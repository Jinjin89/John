# Use Claude Code and Codex together

Run both agents from the command line, then use Codex to draft release notes and Claude Code to review them. The Python pipeline runs in one process and shares a workspace and SQLite ledger.

## Install John and check your logins

From the John checkout, use Python 3.12 or later, `uv`, and already signed-in copies of Claude Code and Codex in the same shell that will run John. The Claude profile uses John's `unset_env` support to remove inherited credentials. The launch flags below were checked with Claude Code 2.1.332 and Codex CLI 0.172.0.

```bash
uv sync --locked
claude auth status --text
codex login status
```

Use `claude --version` and `codex --version` to check which binaries the shell finds. If a login check fails, see [Authentication and fallback](auth-and-fallback.md).

## Create the demo workspace and config

In a fresh demo workspace, create the input and output directory.

```bash
mkdir -p workspace/out
cat > workspace/CHANGELOG_NOTES.txt <<'EOF'
- CSV exports now keep empty columns in their original order.
- The progress indicator updates during large exports.
- Existing export settings still work.
EOF
```

Save the following as `agents.toml` next to `workspace/`. Relative paths resolve from the config file.

```toml
[ledger]
path = ".john/ledger.db"

[workspaces.demo]
path = "./workspace"
max_access = "read_write"

[agents.codex]
kind = "codex"
command = "codex"
args = ["app-server"]
default_access = "read_write"
require_workspace = true

[agents.codex.capabilities]
approvals = "live"
enforced_sandbox = ["read_only", "read_write"]

[agents.codex.options]
preflight_auth = true

[agents.claude]
kind = "claude"
command = "claude"
args = [
    "--restricted", "--strict-mcp-config", "--no-session-persistence", "--max-turns", "4",
    "--tools", "Read,Glob,Grep",
]
default_access = "read_only"
require_workspace = true
unset_env = ["ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL"]

[agents.claude.capabilities]
approvals = "live"
enforced_sandbox = ["read_only"]
```

With `profile_dir` omitted, Codex uses your existing Codex home (normally `~/.codex`). John requests and verifies Codex's `read-only` or `workspace-write` sandbox for the requested grant. For writes, John limits the writable root to the workspace. [Codex sandbox presets](https://learn.chatgpt.com/docs/codex-sdk#sandbox-presets)

John rejects workspace access missing from the profile's `enforced_sandbox` declaration, which is empty by default. Declare only the access your launcher enforces. Claude's `--restricted` flag confines built-in file tools to the working directory here and removes command and code execution tools. The `--tools` list limits built-in file access to reads, and `--strict-mcp-config` uses only explicitly provided MCP servers. This profile provides none. If managed hooks or integrations can write to the workspace, enforce read-only access in the launcher before using this declaration. [Claude Code CLI reference](https://code.claude.com/docs/en/cli-reference)

## 1. Run each agent from the CLI

Both commands use stdio and emit runtime events and the final A2A task as newline-delimited JSON. The workspace grant is read-only for these checks.

```bash
.venv/bin/john run --config agents.toml --agent codex --workspace demo:read_only --json \
  "Read CHANGELOG_NOTES.txt and summarize it in one sentence."

.venv/bin/john run --config agents.toml --agent claude --workspace demo:read_only --json \
  "Read CHANGELOG_NOTES.txt and summarize it in one sentence."
```

## 2. Draft and review release notes in Python

Save the following as `pipeline.py`. Codex writes a declared output, which John returns as an A2A artifact containing a workspace-relative path and SHA-256 digest. Claude receives that artifact in the same workspace with read-only access. John validates the review against the JSON Schema before returning a completed task.

```python
import asyncio
import json

from a2a.types import Task, TaskState
from google.protobuf.json_format import MessageToDict

from john import John, OutputSpec, Workspace


VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "approved": {"type": "boolean"},
        "summary": {"type": "string"},
        "issues": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["approved", "summary", "issues"],
    "additionalProperties": False,
}


def require_completed(step: str, task: Task) -> None:
    if task.status.state != TaskState.TASK_STATE_COMPLETED:
        state = TaskState.Name(task.status.state)
        detail = " ".join(
            part.text for part in task.status.message.parts if part.WhichOneof("content") == "text"
        )
        raise RuntimeError(f"{step} ended in {state}. {detail}")


async def main() -> None:
    output = OutputSpec(
        name="release_notes", path="out/release-notes.md", mediaType="text/markdown"
    )
    async with John.open("agents.toml") as j:
        draft = await j.assign(
            "codex",
            f"Read CHANGELOG_NOTES.txt and write concise release notes to {output.path}.",
            workspace=Workspace("demo", "read_write"),
            outputs=[output],
        )
        require_completed("Draft", draft)
        artifact = next(item for item in draft.artifacts if item.artifact_id == output.name)

        review = await j.assign(
            "claude",
            "Review the release notes in the input artifact against CHANGELOG_NOTES.txt. "
            "Identify any omitted or inaccurate details. Return only the JSON verdict.",
            workspace=Workspace("demo", "read_only"),
            inputs=[artifact],
            parent=("codex", draft.id),
            schema=VERDICT_SCHEMA,
        )
        require_completed("Review", review)
        verdict = MessageToDict(review.artifacts[0].parts[0].data)

        runs = await j.ledger.list_runs(task_id=draft.id)
        runs += await j.ledger.list_runs(task_id=review.id)
        lineage = [
            {key: run[key] for key in ("agent_name", "run_id", "parent_run_id")} for run in runs
        ]
        print(json.dumps({"verdict": verdict, "lineage": lineage}, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
```

```bash
.venv/bin/python pipeline.py
```

The script prints the validated verdict and the SQLite run IDs. The review's `parent_run_id` points to the draft's `run_id`.

## 3. Serve Claude over A2A HTTP (optional)

Let the script exit or close any embedded `John` instance using this ledger before serving. John holds an exclusive `.lock` file for the ledger while open.

```bash
.venv/bin/john serve --config agents.toml --agent claude --host 127.0.0.1 --port 8080
```

From a second terminal, fetch the published A2A v1.0 Agent Card.

```bash
curl -s http://127.0.0.1:8080/.well-known/agent-card.json | python3 -m json.tool
```

## Troubleshooting

| Symptom | What to do |
| --- | --- |
| Ledger is busy (`BusyError`) | Exit the other embedded session or `john serve` using this ledger. |
| Declared output already exists | Choose a fresh relative path in `OutputSpec` for each run. |
| `auth_required` | Check which login and environment the agent is using. [Authentication and fallback](auth-and-fallback.md) |
