"""Internal helpers shared by the stdio session drivers."""

from __future__ import annotations

import asyncio
import json
import math
from collections.abc import Awaitable, Callable, Iterator
from contextlib import suppress
from dataclasses import dataclass, field
from itertools import count
from pathlib import Path
from typing import Any

from john.transport import Frame, redact_text
from john.types import (
    AgentProfile,
    ApprovalDecision,
    ApprovalPolicy,
    ApprovalRequest,
    EventSink,
    OtherEvent,
    RuntimeEvent,
    TransportError,
    TurnRequest,
    UnsupportedError,
    WorkspaceAccess,
    WorkspaceViolationError,
)


def _object(value: Any) -> Frame:
    return value if isinstance(value, dict) else {}


def _string(value: Any, default: str = "") -> str:
    return value if isinstance(value, str) else default


def _integer(value: Any) -> int | None:
    return value if type(value) is int and value >= 0 else None


def _number(value: Any) -> float | None:
    if type(value) in (int, float):
        with suppress(OverflowError):
            number = float(value)
            return number if math.isfinite(number) and number >= 0 else None
    return None


def _safe_payload(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            redact_text(key): (
                "<redacted>"
                if key.lower()
                .replace("-", "")
                .replace("_", "")
                .endswith(
                    (
                        "apikey",
                        "accesskey",
                        "privatekey",
                        "token",
                        "secret",
                        "password",
                        "authorization",
                    )
                )
                else _safe_payload(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_safe_payload(item) for item in value]
    return redact_text(value) if isinstance(value, str) else value


def _init_timeout(profile: AgentProfile) -> float:
    seconds = _number(profile.options.get("init_timeout_s", 30))
    if not seconds:
        raise UnsupportedError("init_timeout_s must be a positive finite number")
    return seconds


def _ignored_options(profile: AgentProfile, supported: set[str]) -> dict[str, str]:
    ignored = {key: "unsupported option" for key in profile.options.keys() - supported}
    if profile.profile_dir is not None:
        ignored["profile_dir"] = "configure the harness using its command, args, or environment"
    return ignored


def _check_scope(profile: AgentProfile, workspace: Path | None, access: WorkspaceAccess) -> None:
    # The operator's launcher supplies isolation. A subprocess cwd alone has no sandbox semantics.
    if workspace is not None and access not in profile.capabilities.enforced_sandbox:
        raise UnsupportedError("The launcher does not declare the requested workspace scope")


def _turn_prompt(request: TurnRequest, workspace: Path | None, access: WorkspaceAccess) -> str:
    if request.continuation is not None:
        raise UnsupportedError("This harness cannot resume a task-level continuation")
    sections = [request.prompt]
    if request.input_refs or request.outputs:
        if workspace is None:
            raise WorkspaceViolationError("Workspace references require a bound workspace")
        if request.outputs and access != "read_write":
            raise WorkspaceViolationError("Declared outputs require a writable workspace")
        root = workspace.resolve()
        for ref in (*request.input_refs, *request.outputs):
            if any(part in ("", ".", "..") for part in ref.path.split("/")) or "\x00" in ref.path:
                raise WorkspaceViolationError("Expected a clean relative workspace path")
            try:
                within_root = (root / ref.path).resolve().is_relative_to(root)
            except (OSError, RuntimeError, ValueError):
                raise WorkspaceViolationError("Could not resolve the workspace path") from None
            if not within_root:
                raise WorkspaceViolationError("The workspace path leaves the authorized root")
        if request.input_refs:
            refs = [ref.model_dump(by_alias=True, mode="json") for ref in request.input_refs]
            sections.append(
                "Read these inputs at their workspace-relative paths:\n" + json.dumps(refs)
            )
        if request.outputs:
            outputs = [spec.model_dump(by_alias=True, mode="json") for spec in request.outputs]
            sections.append(
                "Write the declared outputs at these workspace-relative paths:\n"
                + json.dumps(outputs)
            )
    if request.schema_def is not None:
        sections.append("Return JSON conforming to this schema:\n" + json.dumps(request.schema_def))
    return "\n\n".join(sections)


def _consume_task_exception(task: asyncio.Future[Any]) -> None:
    if not task.cancelled():
        task.exception()


@dataclass(slots=True)
class _Turn:
    """Share event sequencing and the approval/cancel boundary between stdio sessions."""

    request: TurnRequest
    sink: EventSink
    policy: ApprovalPolicy
    include_raw: bool
    seq: Iterator[int] = field(default_factory=count)
    stopped: asyncio.Event = field(default_factory=asyncio.Event)
    done: asyncio.Event = field(default_factory=asyncio.Event)
    dispatched: bool = False
    cancel_requested: bool = False
    interrupt: asyncio.Task[None] | None = None

    @property
    def run_id(self) -> str:
        return self.request.run_id

    async def emit(self, event: Callable[..., RuntimeEvent], **fields: Any) -> None:
        await self.sink(event(run_id=self.run_id, seq=next(self.seq), **fields))

    async def other(self, native_type: str, payload: Frame) -> None:
        await self.emit(
            OtherEvent,
            native_type=redact_text(native_type),
            raw=_safe_payload(payload) if self.include_raw else None,
            redacted=True,
        )

    def request_cancel(self, send: Callable[[], Awaitable[None]]) -> None:
        if self.cancel_requested or self.stopped.is_set():
            return
        self.cancel_requested = True
        if not self.dispatched:
            self.stopped.set()
            return

        async def signal() -> None:
            try:
                with suppress(TransportError):
                    if not self.stopped.is_set():
                        await send()
            finally:
                self.stopped.set()

        self.interrupt = asyncio.create_task(signal(), name="driver interrupt")
        self.interrupt.add_done_callback(_consume_task_exception)

    async def finish(self) -> None:
        self.stopped.set()
        try:
            if self.interrupt is not None:
                self.interrupt.cancel()
                await asyncio.gather(self.interrupt, return_exceptions=True)
        finally:
            self.done.set()

    async def decide(self, permission: ApprovalRequest) -> ApprovalDecision:
        if self.cancel_requested or self.stopped.is_set():
            await self.stopped.wait()
            return ApprovalDecision(outcome="cancelled")

        async def invoke() -> ApprovalDecision:
            return await self.policy(permission)

        decision = asyncio.create_task(invoke())
        stop = asyncio.create_task(self.stopped.wait())
        decision.add_done_callback(_consume_task_exception)
        try:
            await asyncio.wait((decision, stop), return_when=asyncio.FIRST_COMPLETED)
            if self.cancel_requested or self.stopped.is_set():
                # The wire interrupt precedes cancellation replies for pending permissions.
                await self.stopped.wait()
                return ApprovalDecision(outcome="cancelled")
            if not decision.cancelled():
                with suppress(Exception):
                    value = decision.result()
                    if isinstance(value, ApprovalDecision):
                        return value
            return ApprovalDecision(reason="Denied by John policy")
        finally:
            stop.cancel()
            decision.cancel()
