# John

**John** is a headless Python control plane that runs general-purpose agent harnesses (`codex`, `claude`, `agy` / `hermes` via ACP) and custom in-process Python specialists (e.g. `deepagents` / LangGraph) behind one **A2A v1.0** contract — with **no GUI, no TUI, and no PTY screen-scraping**.

## Structure

John separates **inter-agent collaboration** (A2A v1.0 outside) from **local harness execution** (native headless protocols inside):

```text
User / Python Script / CLI / External A2A Client
  │  One outer contract: A2A v1.0 (AgentCard · Task · Message · Artifact · INPUT_REQUIRED)
  ▼
John Coordinator  (in-process DefaultRequestHandler by default; opt-in `john serve` HTTP/SSE)
  ├── SQLite WAL Store (A2A TaskStore + run ledger + idempotency + crash recovery)
  ├── Workspace Extension (dir_fd / O_NOFOLLOW + SHA-256 artifact refs, no base64)
  └── Local Approval Broker (deny-default tool permissions stay inside each bridge)
        │
        ├── john/drivers/codex.py      → `codex app-server` (JSON-RPC stdio)
        ├── john/drivers/acp.py        → ACP v1 (`agy acp`, Docker `hermes acp`, etc.)
        ├── john/drivers/claude.py     → `claude -p --input/output-format stream-json`
        ├── john/drivers/inprocess.py  → In-process async Python specialists (`INPUT_REQUIRED`)
        └── Remote A2A peers           → Direct A2A v1.0 JSON-RPC client (no bridge)
```

### Package Layout

```text
john/
  types.py        Frozen internal SPI (events, capabilities, approvals, errors — no backend names)
  workspace.py    Scoped workspace grants, dir_fd/O_NOFOLLOW file opens, SHA-256 artifact refs
  ledger.py       Unified SQLite WAL store (A2A VersionedTaskStore + run journal + exclusive lock)
  transport.py    Shared StdioTransport + JsonRpcPeer (bounded queues/stderr, process-group reaper)
  runtime.py      Single-turn exclusivity, control-channel ApprovalBroker, JSON Schema validation
  bridge.py       A2A v1.0 AgentExecutor mapping Tasks/Artifacts/INPUT_REQUIRED to Driver sessions
  coordinator.py  Embedded zero-daemon John facade, agents.toml loader, HTTP app, remote A2A client
  cli.py          Headless CLI (`john run`, `john serve` — loopback-only without auth)
  drivers/        Protocol-family drivers (`codex.py`, `acp.py`, `claude.py`, `inprocess.py`)
```

**Core invariants:**
- **Control plane vs. data plane:** Harnesses keep their own LLM loop, tools, sandbox, auth, and compaction; John never calls an LLM directly and never translates private chat context (handoff uses real workspace files + SHA-256 refs).
- **Tool approvals ≠ `INPUT_REQUIRED`:** Mid-turn tool permissions are resolved inside the bridge by local async policy (deny by default); genuine task-level clarifications use A2A `TASK_STATE_INPUT_REQUIRED` and resume cleanly via `reply()`.
- **Fail-closed defaults:** `Capabilities.enforced_sandbox` defaults to empty, `john serve` refuses unauthenticated non-loopback binds, and core modules contain zero backend-specific branching (enforced in `tests/test_arch_invariants.py`).

## Quick Start

### 1. Install (Python 3.12 + `uv`)

```bash
uv sync
```

### 2. Configure `agents.toml`

```toml
[ledger]
path = ".john/ledger.db"

[workspaces.project]
path = "/abs/path/to/workspace"
max_access = "read_write"

[agents.codex]
kind = "codex"
description = "Generalist Codex app-server agent"
command = "codex"
args = ["app-server"]
default_access = "read_write"
require_workspace = true

[agents.codex.capabilities]
approvals = "live"
enforced_sandbox = ["read_only", "read_write"]

[agents.qc]
kind = "inprocess"
description = "Read-only checkpointed QC specialist"
default_access = "read_only"
require_workspace = true

[agents.qc.capabilities]
approvals = "live"
task_questions = true
enforced_sandbox = ["read_only"]

[agents.qc.options]
handler = "invoice-qc"
```

### 3. Run From Python (Zero Daemons)

```python
import asyncio
from a2a.types import TaskState
from john import John, OutputSpec, Workspace

async def main() -> None:
    async with John.open("agents.toml") as j:
        # Step 1: Generalist produces a declared workspace artifact
        prep = await j.assign(
            "codex",
            "Normalize vendor exports into out/invoices.csv.",
            workspace=Workspace("project", "read_write"),
            outputs=[OutputSpec(name="normalized", path="out/invoices.csv", mediaType="text/csv")],
        )
        if prep.status.state != TaskState.TASK_STATE_COMPLETED:
            return

        # Step 2: Specialist verifies SHA-256 ref, may pause at INPUT_REQUIRED, then completes
        artifact = next(a for a in prep.artifacts if a.artifact_id == "normalized")
        qc = await j.assign(
            "qc",
            "Verify normalized invoices.",
            workspace=Workspace("project", "read_only"),
            inputs=[artifact],
            parent=("codex", prep.id),
        )
        if qc.status.state == TaskState.TASK_STATE_INPUT_REQUIRED:
            qc = await j.reply("qc", qc, "0.80")
        print(TaskState.Name(qc.status.state), qc.artifacts)

asyncio.run(main())
```

### 4. Run From CLI

```bash
# Single pipeable run (text output or NDJSON event stream)
echo "Summarize the repository." | .venv/bin/john run --config agents.toml --agent codex
echo "Summarize the repository." | .venv/bin/john run --config agents.toml --agent codex --json

# Expose any local agent as an A2A v1.0 JSON-RPC/SSE endpoint (loopback default)
.venv/bin/john serve --config agents.toml --agent qc --host 127.0.0.1 --port 8080
```

### 5. Test & Lint

```bash
.venv/bin/pytest
.venv/bin/ruff check . && .venv/bin/ruff format --check .
```
