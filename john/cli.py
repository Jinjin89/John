"""Command line entry points for running and serving agents."""

from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
import sys
from collections.abc import Sequence
from typing import Any

import httpx
import uvicorn
from a2a.types import Task, TaskState
from a2a.utils.errors import A2AError
from google.protobuf.json_format import MessageToDict

from john.coordinator import John, Workspace, require_loopback_host
from john.transport import redact_text
from john.types import ErrorEvent, JohnError, OutputSpec, RuntimeEvent, StartedEvent


def _workspace(value: str) -> Workspace:
    root, separator, access = value.rpartition(":")
    if not separator or not root or access not in {"read_only", "read_write"}:
        raise argparse.ArgumentTypeError("expected ROOT:read_only or ROOT:read_write")
    return Workspace(root, "read_only" if access == "read_only" else "read_write")


def _output(value: str) -> OutputSpec:
    fields = value.split(":", maxsplit=2)
    if len(fields) != 3 or not all(fields):
        raise argparse.ArgumentTypeError("expected NAME:PATH:MEDIA_TYPE")
    name, path, media_type = fields
    return OutputSpec(name=name, path=path, media_type=media_type)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="john")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Send a task to an agent")
    serve = commands.add_parser("serve", help="Serve a local agent over A2A")
    for command in (run, serve):
        command.add_argument("--config", default="agents.toml")
        command.add_argument("--agent", required=True)
    run.add_argument("--workspace", type=_workspace, metavar="ROOT:ACCESS")
    run.add_argument("--output", type=_output, action="append", metavar="NAME:PATH:MEDIA_TYPE")
    run.add_argument(
        "--json", action="store_true", help="Emit NDJSON events and the final A2A task"
    )
    run.add_argument("prompt", nargs="?", help="Prompt text, or read from stdin")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8080)
    return parser


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _answer(task: Task) -> str:
    parts = [part for artifact in task.artifacts for part in artifact.parts]
    if not parts:
        parts = list(task.status.message.parts)
    lines = []
    for part in parts:
        match part.WhichOneof("content"):
            case "text":
                lines.append(part.text)
            case "url":
                lines.append(part.url)
            case "data":
                lines.append(_json(MessageToDict(part.data)))
            case _:
                lines.append(_json(MessageToDict(part)))
    return "\n".join(lines)


async def _run(args: argparse.Namespace) -> int:
    prompt = args.prompt if args.prompt is not None else sys.stdin.read()

    async def observe(event: RuntimeEvent) -> None:
        if args.json:
            print(event.model_dump_json(), flush=True)
        elif isinstance(event, StartedEvent):
            print(f"Started run {event.run_id}", file=sys.stderr)
        elif isinstance(event, ErrorEvent):
            print(redact_text(event.message), file=sys.stderr)

    async with John.open(args.config) as john:
        task = await john.assign(
            args.agent,
            prompt,
            workspace=args.workspace,
            outputs=args.output,
            on_event=observe if john.is_local(args.agent) else None,
        )
    if args.json:
        print(_json(MessageToDict(task)), flush=True)
    if task.status.state == TaskState.TASK_STATE_COMPLETED:
        if not args.json and (answer := _answer(task)):
            print(answer)
        return 0
    state = TaskState.Name(task.status.state)
    if detail := _answer(task):
        state = f"{state} {redact_text(detail)}"
    print(state, file=sys.stderr)
    return 1


async def _serve(args: argparse.Namespace) -> int:
    require_loopback_host(args.host)
    advertised_host = f"[{args.host}]" if ":" in args.host else args.host
    url = f"http://{advertised_host}:{args.port}/a2a"
    async with John.open(args.config) as john:
        app = john.build_http_app(args.agent, url=url)
        server = uvicorn.Server(uvicorn.Config(app, host=args.host, port=args.port))
        await server.serve()
        return 0 if server.started else 1


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return asyncio.run(_run(args) if args.command == "run" else _serve(args))
    except KeyboardInterrupt:
        return 130
    except (A2AError, JohnError, httpx.HTTPError, OSError, ValueError, sqlite3.Error) as error:
        print(redact_text(str(error)), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
