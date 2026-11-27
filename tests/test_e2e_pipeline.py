"""Integrated end-to-end pipeline tests combining default drivers, A2A, and SQLite recovery."""

from __future__ import annotations

import asyncio
import json
import socket
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from a2a.types import TaskState
from google.protobuf.json_format import MessageToDict

from john import (
    AgentProfile,
    ApprovalDecision,
    ApprovalRequest,
    Capabilities,
    John,
    OutputSpec,
    Workspace,
    WorkspaceGrant,
)
from john.ledger import SqliteLedger
from john.types import WORKSPACE_REF_MEDIA_TYPE

_ACP_WRITER_SCRIPT = r"""
import json, pathlib, sys

cwd = None
for raw in sys.stdin:
    msg = json.loads(raw)
    mid = msg.get("id")
    method = msg.get("method")
    params = msg.get("params") or {}
    if method == "initialize":
        resp = {
            "jsonrpc": "2.0",
            "id": mid,
            "result": {
                "protocolVersion": 1,
                "agentInfo": {"name": "mock-acp-writer", "version": "0.1.0"},
                "agentCapabilities": {"loadSession": False},
            },
        }
        sys.stdout.write(json.dumps(resp) + "\n")
        sys.stdout.flush()
    elif method == "session/new":
        cwd = pathlib.Path(params["cwd"])
        resp = {"jsonrpc": "2.0", "id": mid, "result": {"sessionId": "acp-sess-1"}}
        sys.stdout.write(json.dumps(resp) + "\n")
        sys.stdout.flush()
    elif method == "session/prompt":
        out = cwd / "out" / "invoices.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            "invoice_id,score\nINV-001,0.12\nINV-002,0.85\nINV-003,0.92\n",
            encoding="utf-8",
        )
        note = {
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "sessionId": "acp-sess-1",
                "update": {
                    "sessionUpdate": "agent_message_chunk",
                    "content": {"type": "text", "text": "Normalized invoices to out/invoices.csv."},
                },
            },
        }
        sys.stdout.write(json.dumps(note) + "\n")
        resp = {"jsonrpc": "2.0", "id": mid, "result": {"stopReason": "end_turn"}}
        sys.stdout.write(json.dumps(resp) + "\n")
        sys.stdout.flush()
"""


@asynccontextmanager
async def _serve_loopback(john: John, agent: str) -> AsyncIterator[str]:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        url = f"http://127.0.0.1:{listener.getsockname()[1]}/rpc"
        app = john.build_http_app(agent, url=url)
        server = uvicorn.Server(
            uvicorn.Config(app, log_level="error", access_log=False, timeout_graceful_shutdown=3)
        )
        serving = asyncio.create_task(server.serve(sockets=[listener]))
        try:
            async with asyncio.timeout(5):
                while not server.started:
                    if serving.done():
                        serving.result()
                        raise RuntimeError("Server exited before startup")
                    await asyncio.sleep(0.01)
            yield url
        finally:
            server.should_exit = True
            await asyncio.wait_for(serving, 5)


async def test_e2e_generalist_acp_to_specialist_inprocess_with_restart(tmp_path: Path) -> None:
    """Generalist ACP writes CSV -> SHA-256 ref -> QC pauses, survives restart, and completes."""
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    script_path = tmp_path / "acp_writer.py"
    script_path.write_text(_ACP_WRITER_SCRIPT, encoding="utf-8")

    config_path = tmp_path / "agents.toml"
    config_path.write_text(
        f"""
[ledger]
path = "state/ledger.db"

[workspaces.billing]
path = {json.dumps(str(workspace_dir))}
max_access = "read_write"

[agents.generalist]
kind = "acp"
description = "Generalist ACP worker"
command = {json.dumps(sys.executable)}
args = [{json.dumps(str(script_path))}]
default_access = "read_write"
require_workspace = true

[agents.generalist.capabilities]
approvals = "live"
enforced_sandbox = ["read_only", "read_write"]

[agents.qc]
kind = "inprocess"
description = "Read-only invoice QC specialist"
default_access = "read_only"
require_workspace = true

[agents.qc.capabilities]
approvals = "live"
task_questions = true
enforced_sandbox = ["read_only"]

[agents.qc.options]
handler = "invoice-qc"
""",
        encoding="utf-8",
    )

    schema = {
        "type": "object",
        "properties": {
            "cutoff": {"type": "number"},
            "flagged": {"type": "array", "items": {"type": "string"}},
            "total": {"type": "integer"},
            "passed": {"type": "integer"},
        },
        "required": ["cutoff", "flagged", "total", "passed"],
    }

    # Phase 1: Generalist writes CSV; Specialist reads verified ref & pauses at INPUT_REQUIRED.
    async with John.open(config_path) as john:
        assert john.list_cards().keys() == {"generalist", "qc"}
        prep = await john.assign(
            "generalist",
            "Normalize vendor exports to out/invoices.csv.",
            workspace=Workspace("billing", "read_write"),
            outputs=[OutputSpec(name="normalized", path="out/invoices.csv", mediaType="text/csv")],
        )
        assert prep.status.state == TaskState.TASK_STATE_COMPLETED
        artifact = next(a for a in prep.artifacts if a.artifact_id == "normalized")
        assert artifact.parts[0].media_type == WORKSPACE_REF_MEDIA_TYPE

        paused = await john.assign(
            "qc",
            "Check normalized invoices for anomalies.",
            workspace=Workspace("billing", "read_only"),
            inputs=[artifact],
            parent=("generalist", prep.id),
            schema=schema,
        )
        assert paused.status.state == TaskState.TASK_STATE_INPUT_REQUIRED
        assert "cutoff" in paused.status.message.parts[0].text.lower()

    # Phase 2: Reopen John from disk (simulated process restart) and resume paused specialist.
    async with John.open(config_path) as restarted:
        recovered = await restarted.get("qc", paused.id)
        assert recovered.status.state == TaskState.TASK_STATE_INPUT_REQUIRED

        completed = await restarted.reply("qc", recovered, "0.80")
        assert completed.status.state == TaskState.TASK_STATE_COMPLETED
        assert len(completed.artifacts) == 1
        verdict = MessageToDict(completed.artifacts[0].parts[0].data)
        assert verdict["cutoff"] == 0.8
        assert verdict["flagged"] == ["INV-002", "INV-003"]
        assert verdict["total"] == 3
        assert verdict["passed"] == 1

        # Verify lineage recorded in SQLite across generalist -> qc pause -> qc resume.
        prep_runs = await restarted.ledger.list_runs(task_id=prep.id)
        qc_runs = await restarted.ledger.list_runs(task_id=paused.id)
        assert len(prep_runs) == 1
        assert len(qc_runs) == 2
        assert qc_runs[0]["parent_run_id"] == prep_runs[0]["run_id"]
        assert qc_runs[1]["parent_run_id"] == qc_runs[0]["run_id"]


async def test_e2e_single_deduplicated_approval_events_in_ledger(tmp_path: Path) -> None:
    """In-process specialist approval produces exactly one requested + one decided event."""
    db_path = tmp_path / "ledger.db"
    ledger = await SqliteLedger.open(db_path)
    profile = AgentProfile(
        name="echo",
        kind="inprocess",
        description="Approval echo specialist",
        capabilities=Capabilities(approvals="live", enforced_sandbox=frozenset({"read_only"})),
        options={"handler": "echo", "approval_action": "shell.exec"},
    )

    seen_requests: list[ApprovalRequest] = []

    async def allow_once(req: ApprovalRequest) -> ApprovalDecision:
        seen_requests.append(req)
        return ApprovalDecision(outcome="allow_once", reason="approved in test")

    async with John(profiles={"echo": profile}, workspaces=(), ledger=ledger) as john:
        task = await john.assign("echo", "ping", policy=allow_once)
        assert task.status.state == TaskState.TASK_STATE_COMPLETED
        assert len(seen_requests) == 1
        runs = await john.ledger.list_runs(task_id=task.id)
        assert len(runs) == 1
        events = await john.ledger.get_run_events(runs[0]["run_id"])
        approval_events = [e for e in events if e.kind == "approval"]
        assert [(e.phase, e.outcome) for e in approval_events] == [
            ("requested", None),
            ("decided", "allow_once"),
        ]


async def test_e2e_served_specialist_over_real_a2a_http(tmp_path: Path) -> None:
    """Expose built-in invoice-qc specialist over A2A v1.0 HTTP and complete INPUT_REQUIRED flow."""
    server_ledger = await SqliteLedger.open(tmp_path / "server.db")
    qc_profile = AgentProfile(
        name="qc",
        kind="inprocess",
        description="Served QC specialist",
        capabilities=Capabilities(task_questions=True, enforced_sandbox=frozenset({"read_only"})),
        options={"handler": "invoice-qc"},
    )
    server_john = John(
        profiles={"qc": qc_profile},
        workspaces=(WorkspaceGrant(root_id="w", host_path=tmp_path),),
        ledger=server_ledger,
    )
    async with _serve_loopback(server_john, "qc") as url:
        client_ledger = await SqliteLedger.open(tmp_path / "client.db")
        remote_profile = AgentProfile(name="remote-qc", kind="remote", url=url)
        async with John(
            profiles={"remote-qc": remote_profile}, workspaces=(), ledger=client_ledger
        ) as client:
            csv_input = "invoice_id,score\nA-1,0.2\nA-2,0.9\n"
            paused = await client.assign("remote-qc", csv_input)
            assert paused.status.state == TaskState.TASK_STATE_INPUT_REQUIRED
            done = await client.reply("remote-qc", paused, "0.5")
            assert done.status.state == TaskState.TASK_STATE_COMPLETED
            result = MessageToDict(done.artifacts[0].parts[0].data)
            assert result["flagged"] == ["A-2"]
            assert result["total"] == 2
