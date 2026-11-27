# Authentication inheritance and automatic auth fallback

## Use your existing logins

Run John as the user who signed in to each CLI. Claude Code can use the saved OAuth login for that user, with its normal `~/.claude` configuration and credential storage. In print mode, `ANTHROPIC_AUTH_TOKEN` or `ANTHROPIC_API_KEY` can take precedence over OAuth. `ANTHROPIC_BASE_URL` can send requests to a different endpoint. Restricted mode limits which settings files Claude loads. [Claude Code authentication precedence](https://code.claude.com/docs/en/authentication#authentication-precedence)

Codex uses the existing `CODEX_HOME` when `profile_dir` is omitted and defaults to `~/.codex` when `CODEX_HOME` is unset. The cached login can live in that directory or the system credential store. Use `codex login status` in the launch environment and set `options.preflight_auth = true` for John's account check before a turn. [Codex authentication](https://learn.chatgpt.com/docs/auth?surface=cli)

Set `unset_env` on a subprocess profile to remove conflicting variables inherited from the shell. Use the Claude list below to clear stale credentials and the gateway URL. For Codex, use `unset_env = ["OPENAI_API_KEY"]` if an inherited API key conflicts with the intended login or provider. John removes those variables from the child environment before applying the explicit profile `env`, so you can still supply a profile-specific override.

## Isolate alternate credentials

For a separate Codex API-key profile, set `profile_dir` to an absolute directory disjoint from the workspace. This sets the child process's `CODEX_HOME`. With the API key already exported, sign in to that same directory. [Codex CLI login reference](https://learn.chatgpt.com/docs/cli/reference#codex-login)

```bash
printenv OPENAI_API_KEY | CODEX_HOME=/absolute/private/codex-api codex login --with-api-key
```

For a Claude API-key or gateway profile, pass `--settings /absolute/private/claude-gateway.json` in `args`. Keep the settings outside the workspace and source control. For a bearer-token gateway, use a settings file like the one below with your own URL and token. For direct Anthropic API access, set `ANTHROPIC_API_KEY` instead of the gateway token and URL. [Claude Code settings precedence](https://code.claude.com/docs/en/settings#settings-precedence)

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "https://gateway.example.com",
    "ANTHROPIC_AUTH_TOKEN": "your-gateway-token"
  }
}
```

## How John identifies an auth failure

| Driver | Native signal John uses |
| --- | --- |
| Claude Code | A native initialization or API error indicating HTTP 401, `authentication_error`, `invalid_api_key`, or an explicit login or account verification failure. A generic 403 permission denial stays a permissions error. |
| Codex | A missing account when `account/read` requires OpenAI auth and preflight is enabled, or an auth error in a setup RPC or the final native turn error, such as `unauthorized` or HTTP 401. |
| ACP | JSON-RPC error code `-32000` during session setup or `session/prompt`. |

The drivers surface a setup auth failure as `AuthRequiredError` and a turn auth failure as a failed result with `reason="auth_required"`. When the auth failure ends the A2A task, John returns `TASK_STATE_FAILED` with an `auth_required` status message and `task.metadata[LIFECYCLE_EXT_URI]["auth_required"] = True`. Import `LIFECYCLE_EXT_URI` from `john` to read the flag.

## Configure automatic fallback

Use a John build that applies `unset_env` and supports the auth fallback lifecycle. With an empty `fallback_agents` list, a profile returns its own outcome. This example tries the signed-in Claude profile, then the Claude gateway profile, then Codex. Use the demo workspace from the quick start and point the gateway profile at your private settings file. [Claude Code and Codex quick start](quickstart-claude-codex.md)

```toml
[ledger]
path = ".john/ledger.db"

[workspaces.demo]
path = "./workspace"
max_access = "read_write"

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
fallback_agents = ["claude-gateway", "codex"]

[agents.claude.capabilities]
approvals = "live"
enforced_sandbox = ["read_only"]

[agents.claude-gateway]
kind = "claude"
command = "claude"
args = [
    "--restricted", "--strict-mcp-config", "--no-session-persistence", "--max-turns", "4",
    "--tools", "Read,Glob,Grep",
    "--settings", "/absolute/private/claude-gateway.json",
]
default_access = "read_only"
require_workspace = true
unset_env = ["ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL"]

[agents.claude-gateway.capabilities]
approvals = "live"
enforced_sandbox = ["read_only"]

[agents.codex]
kind = "codex"
command = "codex"
args = ["app-server"]
default_access = "read_write"
require_workspace = true
unset_env = ["OPENAI_API_KEY"]

[agents.codex.capabilities]
approvals = "live"
enforced_sandbox = ["read_only", "read_write"]

[agents.codex.options]
preflight_auth = true
```

### When John tries a fallback

- Fallback applies to an initial assignment. It can follow an `AuthRequiredError` while opening the driver, before dispatch, or a dispatched turn that failed with `auth_required`, recorded zero `ToolEvent`s, and left every declared output absent.
- Any recorded `ToolEvent`, including a read, stops fallback. A non-auth error, `indeterminate` result, cancellation, or expired deadline also stops it. Custom in-process specialists must emit `ToolEvent`s before performing side effects because John uses the event journal and declared outputs to decide whether another attempt is safe.
- John checks the original workspace grant and requested access for each local candidate. Fallback configuration rejects unknown targets, cycles, and remote targets.

### Task IDs and replies

Embedded `assign()` and HTTP/SSE `SendMessage` and `SendStreamingMessage` use one public A2A task ID and context ID across attempts. Continue using the original agent and task ID with `get()`, `cancel()`, and `reply()`. Retrying the same request with the same `message_id` resolves to that task without starting another chain. A `reply()` resumes the profile that owns the checkpoint and never starts fallback.

If a fallback completes the task or pauses for input, `task.metadata[LIFECYCLE_EXT_URI]` contains `fallback_from` and `resolved_agent`. If all eligible attempts fail authentication, the task ends as `TASK_STATE_FAILED` with the `auth_required` flag.
