import asyncio
import copy
import json
import os
import pwd
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from john.drivers.codex import CodexAppServerDriver
from john.types import (
    AgentProfile,
    ApprovalDecision,
    ApprovalOutcome,
    ApprovalRequest,
    AuthRequiredError,
    BusyError,
    Capabilities,
    Driver,
    DriverSession,
    ErrorEvent,
    OtherEvent,
    OutputSpec,
    PlanEvent,
    ResumeUnavailableError,
    RuntimeEvent,
    TextEvent,
    ThoughtEvent,
    ToolEvent,
    TurnRequest,
    UnsupportedError,
    UsageEvent,
    WorkspaceRef,
    WorkspaceViolationError,
    deny_all,
)

_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "codex"
_TRANSCRIPT = [
    json.loads(line) for line in (_FIXTURES / "transcript_blocked.jsonl").read_text().splitlines()
]
_RECORDINGS = {
    row["message"]["id"]: row["message"]["result"]
    for row in _TRANSCRIPT
    if row["direction"] == "server" and "result" in row["message"]
}
_THREAD = _RECORDINGS[2]["thread"]["id"]
_TURN = "test-turn"
_COMMAND = "item/commandExecution/requestApproval"
_FILE = "item/fileChange/requestApproval"

_MOCK = r"""
import json
import os
import sys
import time
from pathlib import Path

from jsonschema import Draft7Validator

script = json.loads(Path(sys.argv[1]).read_text())
schema = json.loads(Path(sys.argv[2]).read_text())
assert Path.cwd() == Path(script['cwd']), 'Wrong subprocess directory'
for key, value in script['env'].items():
    assert os.environ.get(key) == value, key
definitions = schema['definitions']
methods = {
    name: {
        item['properties']['method']['enum'][0] for item in definitions[name]['oneOf']
    }
    for name in ('ClientRequest', 'ClientNotification', 'ServerRequest', 'ServerNotification')
}
client_calls, server_calls, ids = {}, {}, {}
received = []

def validate(value, subschema):
    Draft7Validator({**subschema, 'definitions': definitions}).validate(value)

def check(frame, sender):
    assert 'jsonrpc' not in frame, frame
    validate(frame, {'$ref': '#/definitions/JSONRPCMessage'})
    if sender == 'Client':
        calls, peers = client_calls, server_calls
    else:
        calls, peers = server_calls, client_calls
    if 'method' in frame:
        method = frame['method']
        category = sender + ('Request' if 'id' in frame else 'Notification')
        if method in methods[category]:
            validate(frame, {'$ref': '#/definitions/' + category})
        if 'id' in frame:
            calls[frame['id']] = method
    else:
        method = peers.pop(frame['id'])
        if 'result' in frame and method in schema['x-results']:
            validate(frame['result'], schema['x-results'][method])

for step in script['steps']:
    if 'expect' in step:
        frame = json.loads(input())
        check(frame, 'Client')
        received.append(frame)
        for key, value in step['expect'].items():
            assert key in frame and frame[key] == value, (key, frame, step)
            if key == 'id':
                assert type(frame[key]) is type(value), (key, frame, step)
        if 'capture' in step:
            ids[step['capture']] = frame['id']
    elif 'send' in step:
        frame = step['send'].copy()
        if 'reply_to' in step:
            frame['id'] = ids[step['reply_to']]
        check(frame, 'Server')
        print(json.dumps(frame), flush=True)
    elif 'signal' in step:
        Path(script['marker']).write_text(step['signal'])
    elif 'wait_signal' in step:
        marker = Path(script['marker'])
        while marker.read_text() != step['wait_signal']:
            time.sleep(0.005)

if not script['exit']:
    assert sys.stdin.read() == '', 'Unexpected client frames after the script'
Path(script['log']).write_text(json.dumps(received))
"""


class Events(list[RuntimeEvent]):
    async def __call__(self, event: RuntimeEvent) -> None:
        self.append(event)


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    directory = tmp_path / "workspace"
    directory.mkdir()
    return directory


@dataclass(frozen=True)
class Mock:
    profile: AgentProfile
    marker: Path
    log: Path

    def frames(self) -> list[dict[str, Any]]:
        assert self.log.is_file(), "Mock child did not complete its schema-validated script"
        return json.loads(self.log.read_text())

    async def wait_signal(self, message: str) -> None:
        while not self.marker.exists() or self.marker.read_text() != message:
            await asyncio.sleep(0.005)


def mock_server(
    workspace: Path,
    steps: list[dict[str, Any]],
    *,
    name: str = "server",
    resume: bool = False,
    options: dict[str, Any] | None = None,
    env: dict[str, str] | None = None,
    unset_env: tuple[str, ...] = (),
    expected_env: dict[str, str | None] | None = None,
    profile_dir: Path | None = None,
    exit_after_steps: bool = False,
) -> Mock:
    directory = workspace.parent / name
    directory.mkdir()
    marker, log, script = (
        directory / filename for filename in ("signal", "log.json", "script.json")
    )
    script.write_text(
        json.dumps(
            {
                "steps": steps,
                "cwd": str(workspace),
                "env": expected_env or {},
                "marker": str(marker),
                "log": str(log),
                "exit": exit_after_steps,
            }
        )
    )
    profile = AgentProfile(
        name="codex-fixture",
        kind="codex",
        command=sys.executable,
        args=("-u", "-c", _MOCK, str(script), str(_FIXTURES / "schema_summary.json")),
        capabilities=Capabilities(resume=resume),
        options=options or {},
        profile_dir=profile_dir,
        env=env or {},
        unset_env=unset_env,
    )
    return Mock(profile, marker, log)


def expect(method: str, capture: str | None = None, **params: Any) -> dict[str, Any]:
    wire = {"method": method}
    if params:
        wire["params"] = params
    step = {"expect": wire}
    if capture is not None:
        step["capture"] = capture
    return step


def reply(capture: str, result: dict[str, Any]) -> dict[str, Any]:
    return {"send": {"result": result}, "reply_to": capture}


def notify(method: str, **params: Any) -> dict[str, Any]:
    return {"send": {"method": method, "params": params}}


def turn(id_: str = _TURN, status: str = "inProgress", **fields: Any) -> dict[str, Any]:
    return {"id": id_, "status": status, "items": [], **fields}


def initialize() -> list[dict[str, Any]]:
    return [
        expect("initialize", "initialize", clientInfo={"name": "john", "version": "0.1.0"}),
        reply("initialize", _RECORDINGS["initialize"]),
        expect("initialized"),
    ]


def restore_thread_fields(thread: dict[str, Any], workspace: Path) -> None:
    thread["cwd"] = str(workspace)
    if thread.get("historyMode") == "<redacted>":
        thread["historyMode"] = "paginated"
    if thread.get("source") == "<redacted>":
        thread["source"] = "appServer"


def thread_setup(
    workspace: Path,
    *,
    policy: str = "on-request",
    ephemeral: bool = True,
    writable: bool = False,
    resume: bool = False,
    response_changes: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    params = {
        "cwd": str(workspace),
        "sandbox": "workspace-write" if writable else "read-only",
        "approvalPolicy": policy,
        "approvalsReviewer": "user",
    }
    if writable:
        params["config"] = {
            "sandbox_workspace_write": {
                "network_access": False,
                "writable_roots": [],
                "exclude_tmpdir_env_var": True,
                "exclude_slash_tmp": True,
            }
        }
    if resume:
        params.update(threadId=_THREAD, excludeTurns=True)
    else:
        params["ephemeral"] = ephemeral
    response = copy.deepcopy(_RECORDINGS[2])
    response.update(
        cwd=str(workspace), approvalPolicy=policy, runtimeWorkspaceRoots=[str(workspace)]
    )
    restore_thread_fields(response["thread"], workspace)
    response["thread"]["ephemeral"] = ephemeral
    if writable:
        response["sandbox"] = {
            "type": "workspaceWrite",
            "networkAccess": False,
            "excludeTmpdirEnvVar": True,
            "excludeSlashTmp": True,
            "writableRoots": [],
        }
    response.update(response_changes or {})
    return [
        expect("thread/resume" if resume else "thread/start", "thread", **params),
        reply("thread", response),
    ]


def simple_turn(id_: str, text: str) -> list[dict[str, Any]]:
    return [
        expect("turn/start", "run"),
        reply("run", {"turn": turn(id_)}),
        notify(
            "item/completed",
            threadId=_THREAD,
            turnId=id_,
            completedAtMs=1,
            item={"id": "answer", "type": "agentMessage", "text": text},
        ),
        notify("turn/completed", threadId=_THREAD, turn=turn(id_, "completed")),
    ]


async def test_handshake_and_early_notifications_are_schema_checked(workspace: Path) -> None:
    info = {"threadId": _THREAD, "turnId": _TURN}
    command = {
        "id": "cmd",
        "type": "commandExecution",
        "command": "echo ready",
        "commandActions": [],
        "cwd": str(workspace),
        "status": "inProgress",
    }
    usage = {
        "inputTokens": 9,
        "outputTokens": 4,
        "totalTokens": 13,
        "cachedInputTokens": 0,
        "reasoningOutputTokens": 0,
    }
    detail = {"id": "final", "text": "Done.", "type": "agentMessage"}
    steps = [
        *initialize(),
        *thread_setup(workspace),
        expect(
            "turn/start",
            "run",
            threadId=_THREAD,
            input=[{"type": "text", "text": "Check the sample"}],
        ),
        notify("turn/started", threadId=_THREAD, turn=turn()),
        notify("turn/completed", threadId="other-thread", turn=turn(_TURN, "failed")),
        notify("turn/completed", threadId=_THREAD, turn=turn("other-turn", "failed")),
        notify(
            "item/agentMessage/delta",
            threadId=_THREAD,
            turnId="other-turn",
            itemId="x",
            delta="stale",
        ),
        notify("item/agentMessage/delta", **info, itemId="preview", delta="Pre"),
        notify("item/agentMessage/delta", **info, itemId="preview", delta="view"),
        notify(
            "item/completed",
            **info,
            completedAtMs=1,
            item={"id": "preview", "text": "Preview", "type": "agentMessage"},
        ),
        notify(
            "item/reasoning/summaryTextDelta",
            **info,
            itemId="thought",
            summaryIndex=0,
            delta="Check",
        ),
        notify(
            "item/reasoning/textDelta", **info, itemId="thought", contentIndex=0, delta=" details"
        ),
        notify("item/started", **info, item=command, startedAtMs=1),
        notify("item/completed", **info, item={**command, "status": "failed"}, completedAtMs=2),
        notify(
            "turn/plan/updated", **info, plan=[{"step": "Compare rows", "status": "inProgress"}]
        ),
        notify("thread/tokenUsage/updated", **info, tokenUsage={"last": usage, "total": usage}),
        notify("vendor/extras", **info, token="fixture-secret"),
        notify("item/completed", **info, completedAtMs=3, item=detail),
        notify("item/completed", **info, completedAtMs=3, item=detail),
        notify("turn/completed", threadId=_THREAD, turn=turn(_TURN, "completed")),
        reply("run", {"turn": turn()}),
    ]
    mock = mock_server(workspace, steps)
    driver = CodexAppServerDriver()
    assert isinstance(driver, Driver)
    events = Events()
    async with (
        asyncio.timeout(8),
        driver.open(mock.profile, workspace_path=workspace, access="read_only") as session,
    ):
        assert isinstance(session, DriverSession)
        assert session.native_ref == _THREAD
        assert session.capabilities.enforced_sandbox == {"read_only"}
        assert session.capabilities.approvals == "live"
        outcome = await session.run_turn(
            TurnRequest(run_id="run", prompt="Check the sample"), events, deny_all
        )
    assert outcome.result.status == "completed"
    assert outcome.result.text == "Preview\nDone."
    assert outcome.result.usage == {"input_tokens": 9, "output_tokens": 4}
    assert [(event.text, event.is_delta) for event in events if isinstance(event, TextEvent)] == [
        ("Pre", True),
        ("view", True),
        ("Done.", False),
    ]
    assert [event.text for event in events if isinstance(event, ThoughtEvent)] == [
        "Check",
        " details",
    ]
    assert [(event.name, event.phase) for event in events if isinstance(event, ToolEvent)] == [
        ("commandExecution", "started"),
        ("commandExecution", "failed"),
    ]
    assert [event.items for event in events if isinstance(event, PlanEvent)] == [("Compare rows",)]
    assert [event.input_tokens for event in events if isinstance(event, UsageEvent)] == [9]
    (extra,) = [event for event in events if isinstance(event, OtherEvent)]
    assert extra.native_type == "vendor/extras" and extra.raw is None and extra.redacted
    assert "fixture-secret" not in str(events)
    assert events[0].kind == "started" and events[-1].kind == "finished"
    assert [event.seq for event in events] == list(range(len(events)))
    assert {event.run_id for event in events} == {"run"}
    assert [frame["method"] for frame in mock.frames()] == [
        "initialize",
        "initialized",
        "thread/start",
        "turn/start",
    ]


@pytest.mark.parametrize(
    "changes",
    [
        {"sandbox": {"type": "workspaceWrite"}},
        {"sandbox": {"type": "readOnly", "networkAccess": True}},
        {"approvalPolicy": "never"},
        {"approvalsReviewer": "auto_review"},
        {"cwd": "/tmp/other"},
    ],
)
async def test_refuses_unexpected_effective_scope(workspace: Path, changes: dict[str, Any]) -> None:
    mock = mock_server(
        workspace, [*initialize(), *thread_setup(workspace, response_changes=changes)]
    )
    assert mock.profile.capabilities.enforced_sandbox == frozenset()
    with pytest.raises(UnsupportedError):
        async with CodexAppServerDriver().open(
            mock.profile, workspace_path=workspace, access="read_only"
        ):
            pytest.fail("Unverified thread was opened")
    assert mock.frames()[-1]["method"] == "thread/start"


async def test_workspace_write_prompt_schema_and_child_environment(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = workspace.parent / "state"
    state.mkdir()
    monkeypatch.setenv("CODEX_HOME", str(workspace.parent / "parent-state"))
    monkeypatch.setenv("JOHN_CODEX_TEST_PARENT", "parent")
    mock = mock_server(
        workspace,
        [
            *initialize(),
            *thread_setup(workspace, writable=True),
            *simple_turn(_TURN, '{"count": 1}'),
        ],
        env={"CODEX_HOME": "override", "JOHN_CODEX_TEST_PROFILE": "child"},
        profile_dir=state,
        expected_env={
            "CODEX_HOME": str(state),
            "JOHN_CODEX_TEST_PARENT": "parent",
            "JOHN_CODEX_TEST_PROFILE": "child",
        },
    )
    ref = WorkspaceRef(
        root="test", path="out/in.csv", size_bytes=10, sha256="f" * 64, media_type="text/csv"
    )
    spec = OutputSpec(name="report", path="out/report.json", media_type="application/json")
    schema = {"type": "object", "required": ["count"]}
    async with CodexAppServerDriver().open(
        mock.profile, workspace_path=workspace, access="read_write"
    ) as session:
        assert session.capabilities.enforced_sandbox == {"read_write"}
        assert session.capabilities.native_schema
        result = await session.run_turn(
            TurnRequest(
                run_id="work",
                prompt="Report count",
                input_refs=(ref,),
                outputs=(spec,),
                schema_def=schema,
            ),
            Events(),
            deny_all,
        )
    assert result.result.status == "completed"
    params = mock.frames()[-1]["params"]
    assert params["outputSchema"] == schema
    assert '"out/in.csv" (text/csv, 10 bytes, SHA-256 ' in params["input"][0]["text"]
    assert '"report": "out/report.json" (application/json)' in params["input"][0]["text"]


@pytest.mark.parametrize("override", [False, True])
async def test_codex_removes_configured_environment_variables_before_launch(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, override: bool
) -> None:
    inherited = str(workspace / "inherited-state")
    configured = str(workspace.parent / "configured-state")
    monkeypatch.setenv("CODEX_HOME", inherited)
    monkeypatch.setenv("OPENAI_API_KEY", "parent-fixture")
    monkeypatch.setenv("HOME", str(workspace.parent / "home"))
    mock = mock_server(
        workspace,
        [*initialize(), *thread_setup(workspace)],
        unset_env=("CODEX_HOME", "OPENAI_API_KEY"),
        env={"CODEX_HOME": configured} if override else None,
        expected_env={"CODEX_HOME": configured if override else None, "OPENAI_API_KEY": None},
    )
    async with CodexAppServerDriver().open(
        mock.profile, workspace_path=workspace, access="read_only"
    ) as session:
        assert session.native_ref == _THREAD
    mock.frames()
    assert os.environ["CODEX_HOME"] == inherited
    assert os.environ["OPENAI_API_KEY"] == "parent-fixture"


async def test_workspace_write_rejects_extra_writable_roots(workspace: Path) -> None:
    sandbox = {
        "type": "workspaceWrite",
        "networkAccess": False,
        "excludeTmpdirEnvVar": True,
        "excludeSlashTmp": True,
        "writableRoots": [str(workspace.parent)],
    }
    steps = [
        *initialize(),
        *thread_setup(workspace, writable=True, response_changes={"sandbox": sandbox}),
    ]
    mock = mock_server(workspace, steps)
    with pytest.raises(UnsupportedError, match="unscoped"):
        async with CodexAppServerDriver().open(
            mock.profile, workspace_path=workspace, access="read_write"
        ):
            pytest.fail("Opened an unscoped workspace-write session")
    mock.frames()


@pytest.mark.parametrize("linked", [False, True])
async def test_rejects_profile_home_inside_workspace(workspace: Path, linked: bool) -> None:
    state = workspace / "state"
    state.mkdir()
    if linked:
        alias = workspace.parent / "state-alias"
        alias.symlink_to(state, target_is_directory=True)
        state = alias
    profile = AgentProfile(
        name="example", kind="codex", command="missing-executable", profile_dir=state
    )
    with pytest.raises(WorkspaceViolationError):
        async with CodexAppServerDriver().open(
            profile, workspace_path=workspace, access="read_only"
        ):
            pytest.fail("Opened a nested state directory")


@pytest.mark.parametrize("source", ["profile", "inherited"])
@pytest.mark.parametrize("relation", ["nested", "equal", "ancestor", "relative"])
async def test_rejects_effective_codex_home_from_environment(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, source: str, relation: str
) -> None:
    state = {
        "nested": str(workspace / "state"),
        "equal": str(workspace),
        "ancestor": str(workspace.parent),
        "relative": "state",
    }[relation]
    env = {}
    if source == "profile":
        monkeypatch.setenv("CODEX_HOME", str(workspace.parent / "separate-state"))
        env["CODEX_HOME"] = state
    else:
        monkeypatch.setenv("CODEX_HOME", state)
    profile = AgentProfile(name="example", kind="codex", command="missing-executable", env=env)
    with pytest.raises(WorkspaceViolationError, match="separate from the workspace"):
        async with CodexAppServerDriver().open(
            profile, workspace_path=workspace, access="read_only"
        ):
            pytest.fail("Opened an overlapping state directory")


@pytest.mark.parametrize("source", ["default", "empty-override", "unset", "unset-home"])
async def test_rejects_codex_home_fallback_inside_workspace(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, source: str
) -> None:
    env = {}
    unset_env = ()
    if source == "default":
        monkeypatch.delenv("CODEX_HOME", raising=False)
        monkeypatch.setenv("HOME", str(workspace))
    else:
        monkeypatch.setenv("CODEX_HOME", str(workspace.parent / "separate-state"))
        if source == "empty-override":
            env = {"CODEX_HOME": "", "HOME": str(workspace)}
        elif source == "unset":
            unset_env = ("CODEX_HOME",)
            monkeypatch.setenv("HOME", str(workspace))
        else:
            unset_env = ("CODEX_HOME", "HOME")
            monkeypatch.setenv("HOME", str(workspace.parent))
            monkeypatch.setattr(pwd, "getpwuid", lambda _: SimpleNamespace(pw_dir=str(workspace)))
    profile = AgentProfile(
        name="example", kind="codex", command="missing-executable", env=env, unset_env=unset_env
    )
    with pytest.raises(WorkspaceViolationError, match="separate from the workspace"):
        async with CodexAppServerDriver().open(
            profile, workspace_path=workspace, access="read_only"
        ):
            pytest.fail("Opened an overlapping default state directory")


@pytest.mark.parametrize("authenticated", [False, True])
async def test_account_preflight(workspace: Path, authenticated: bool) -> None:
    account = {"type": "apiKey"} if authenticated else None
    steps = [
        *initialize(),
        expect("account/read", "account", refreshToken=False),
        reply("account", {"account": account, "requiresOpenaiAuth": True}),
    ]
    if authenticated:
        steps.extend(thread_setup(workspace))
    mock = mock_server(workspace, steps, options={"preflight_auth": True})
    context = CodexAppServerDriver().open(
        mock.profile, workspace_path=workspace, access="read_only"
    )
    if authenticated:
        async with context as session:
            assert session.native_ref == _THREAD
    else:
        with pytest.raises(AuthRequiredError):
            async with context:
                pytest.fail("Opened an unauthenticated thread")
    assert mock.frames()[2]["method"] == "account/read"


async def test_multi_turn_and_native_resume(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inherited = str(workspace.parent / "inherited-state")
    monkeypatch.setenv("CODEX_HOME", inherited)
    mock = mock_server(
        workspace,
        [
            *initialize(),
            *thread_setup(workspace, ephemeral=False),
            *simple_turn("turn-one", "First"),
            *simple_turn("turn-two", "Second"),
        ],
        resume=True,
        expected_env={"CODEX_HOME": inherited},
    )
    driver = CodexAppServerDriver()
    async with driver.open(mock.profile, workspace_path=workspace, access="read_only") as session:
        for i, text in enumerate(("First", "Second")):
            result = await session.run_turn(
                TurnRequest(run_id=f"turn-{i}", prompt=text), Events(), deny_all
            )
            assert result.result.text == text and result.result.native_session_ref == _THREAD
        assert session.capabilities.resume
    assert [f["params"]["threadId"] for f in mock.frames()[-2:]] == [_THREAD, _THREAD]

    resumed = mock_server(
        workspace,
        [
            *initialize(),
            *thread_setup(workspace, ephemeral=False, resume=True),
            *simple_turn("turn-three", "Resumed"),
        ],
        name="resumed",
        resume=True,
    )
    async with driver.open(
        resumed.profile, workspace_path=workspace, access="read_only", resume_ref=_THREAD
    ) as session:
        result = await session.run_turn(
            TurnRequest(run_id="third", prompt="Continue"), Events(), deny_all
        )
    assert result.result.text == "Resumed"
    assert resumed.frames()[2]["method"] == "thread/resume"


async def test_resume_unavailable_before_launch_and_on_rpc_error(workspace: Path) -> None:
    profile = AgentProfile(name="no-resume", kind="codex", command="missing-executable")
    driver = CodexAppServerDriver()
    with pytest.raises(ResumeUnavailableError):
        async with driver.open(
            profile, workspace_path=workspace, access="read_only", resume_ref=_THREAD
        ):
            pytest.fail("Opened an ephemeral profile for resume")

    mock = mock_server(
        workspace,
        [
            *initialize(),
            expect("thread/resume", "thread"),
            {
                "send": {"error": {"code": -32000, "message": "Missing thread"}},
                "reply_to": "thread",
            },
        ],
        resume=True,
    )
    with pytest.raises(ResumeUnavailableError, match="RPC -32000"):
        async with driver.open(
            mock.profile, workspace_path=workspace, access="read_only", resume_ref=_THREAD
        ):
            pytest.fail("Opened a missing persisted thread")
    mock.frames()


@pytest.mark.parametrize(
    ("code", "message"),
    [(401, "Request rejected"), (-32000, "Login required"), (-32000, "Access token has expired")],
)
async def test_codex_resume_preserves_authentication_failures(
    workspace: Path, code: int, message: str
) -> None:
    mock = mock_server(
        workspace,
        [
            *initialize(),
            expect("thread/resume", "thread"),
            {"send": {"error": {"code": code, "message": message}}, "reply_to": "thread"},
        ],
        resume=True,
    )
    with pytest.raises(AuthRequiredError, match="authentication is required"):
        async with CodexAppServerDriver().open(
            mock.profile, workspace_path=workspace, access="read_only", resume_ref=_THREAD
        ):
            pytest.fail("Opened a thread without authentication")
    mock.frames()


@pytest.mark.parametrize(
    ("method", "outcome", "decision"),
    [
        (_COMMAND, "allow_once", "accept"),
        (_FILE, "allow_once", "accept"),
        (_COMMAND, "deny", "decline"),
        (_FILE, "deny", "decline"),
        (_COMMAND, "deny_cancel", "cancel"),
        (_FILE, "cancelled", "cancel"),
    ],
)
async def test_live_approval_does_not_block_notifications(
    workspace: Path,
    method: str,
    outcome: ApprovalOutcome,
    decision: str,
) -> None:
    ready = asyncio.Event()
    calls: list[ApprovalRequest] = []
    events = Events()
    request_id = "approval-2" if method == _FILE else 1

    async def sink(event: RuntimeEvent) -> None:
        await events(event)
        if isinstance(event, TextEvent):
            ready.set()

    async def policy(request: ApprovalRequest) -> ApprovalDecision:
        calls.append(request)
        await ready.wait()
        return ApprovalDecision(outcome=outcome)

    args = {
        "threadId": _THREAD,
        "turnId": _TURN,
        "itemId": "tool",
        "startedAtMs": 1,
        "reason": "Approval needed (token=fixture-secret)",
    }
    steps = [
        *initialize(),
        *thread_setup(workspace),
        expect("turn/start", "run"),
        notify("turn/started", threadId=_THREAD, turn=turn()),
        {"send": {"method": method, "id": request_id, "params": args}},
        {"send": {"method": "vendor/unknownRequest", "id": 91}},
        {"expect": {"id": 91, "error": {"code": -32601, "message": "Method not found"}}},
        notify(
            "item/agentMessage/delta",
            threadId=_THREAD,
            turnId=_TURN,
            itemId="answer",
            delta="ready",
        ),
        reply("run", {"turn": turn()}),
        {"expect": {"id": request_id, "result": {"decision": decision}}},
        notify(
            "turn/completed",
            threadId=_THREAD,
            turn=turn(_TURN, "interrupted" if decision == "cancel" else "completed"),
        ),
    ]
    mock = mock_server(workspace, steps)
    async with (
        asyncio.timeout(8),
        CodexAppServerDriver().open(
            mock.profile, workspace_path=workspace, access="read_only"
        ) as session,
    ):
        result = await session.run_turn(
            TurnRequest(run_id="approve", prompt="Test policy"), sink, policy
        )
    (call,) = calls
    assert call.request_id == ("str:approval-2" if method == _FILE else "int:1")
    assert call.action == method and call.run_id == "approve"
    assert call.offered_options == ("accept", "decline", "cancel")
    assert call.args["itemId"] == "tool"
    assert "fixture-secret" not in call.summary + str(call.args)
    assert result.result.status == ("incomplete" if decision == "cancel" else "completed")
    mock.frames()


async def test_approval_ids_preserve_their_wire_types_in_one_turn(workspace: Path) -> None:
    args = {"threadId": _THREAD, "turnId": _TURN, "startedAtMs": 1}
    steps = [
        *initialize(),
        *thread_setup(workspace),
        expect("turn/start", "run"),
        reply("run", {"turn": turn()}),
        {"send": {"method": _COMMAND, "id": 7, "params": {**args, "itemId": "first"}}},
        {"expect": {"id": 7, "result": {"decision": "accept"}}},
        {"send": {"method": _COMMAND, "id": "7", "params": {**args, "itemId": "second"}}},
        {"expect": {"id": "7", "result": {"decision": "decline"}}},
        notify("turn/completed", threadId=_THREAD, turn=turn(_TURN, "completed")),
    ]
    mock = mock_server(workspace, steps)
    calls: list[ApprovalRequest] = []

    async def policy(request: ApprovalRequest) -> ApprovalDecision:
        calls.append(request)
        return ApprovalDecision(outcome="allow_once" if len(calls) == 1 else "deny")

    async with (
        asyncio.timeout(8),
        CodexAppServerDriver().open(
            mock.profile, workspace_path=workspace, access="read_only"
        ) as session,
    ):
        result = await session.run_turn(
            TurnRequest(run_id="approve", prompt="Test"), Events(), policy
        )
    assert result.result.status == "completed"
    assert [(request.run_id, request.request_id) for request in calls] == [
        ("approve", "int:7"),
        ("approve", "str:7"),
    ]
    mock.frames()


@pytest.mark.parametrize(
    ("announce", "terminal_first"), [(False, False), (True, False), (True, True)]
)
async def test_cancel_during_turn_start_drains_matching_terminal(
    workspace: Path, announce: bool, terminal_first: bool
) -> None:
    started = notify("turn/started", threadId=_THREAD, turn=turn())
    ack = reply("run", {"turn": turn()})
    steps = [*initialize(), *thread_setup(workspace), expect("turn/start", "run")]
    if announce:
        steps.append(started)
    steps.append({"signal": "started"})
    if not announce:
        # Wait for the parent to record cancellation, then deliver the turn ID.
        # The signal file is observable while the client waits for its reply.
        steps.append({"wait_signal": "cancelled"})
        steps.extend((ack, started))
    steps.append(expect("turn/interrupt", "interrupt", threadId=_THREAD, turnId=_TURN))
    if announce:
        steps.append(ack)
    terminal = notify("turn/completed", threadId=_THREAD, turn=turn(_TURN, "interrupted"))
    if terminal_first:
        steps.extend((terminal, {"wait_signal": "terminal received"}, reply("interrupt", {})))
    else:
        steps.extend((reply("interrupt", {}), terminal))
    mock = mock_server(workspace, steps)
    driver = CodexAppServerDriver()
    async with (
        asyncio.timeout(8),
        driver.open(mock.profile, workspace_path=workspace, access="read_only") as session,
    ):
        events = Events()
        running = asyncio.create_task(
            session.run_turn(TurnRequest(run_id="cancel", prompt="Hold"), events, deny_all)
        )
        await mock.wait_signal("started")
        await session.cancel("unrelated")
        with pytest.raises(BusyError):
            await session.run_turn(TurnRequest(run_id="overlap", prompt="Hold"), Events(), deny_all)
        try:
            await session.cancel("cancel")
            if not announce:
                mock.marker.write_text("cancelled")
            result = await running
        finally:
            if terminal_first:
                mock.marker.write_text("terminal received")
    assert result.result.status == "cancelled" and result.result.reason == "cancelled_by_caller"
    assert events[-1].kind == "finished" and events[-1].status == "cancelled"
    assert [
        frame["params"] for frame in mock.frames() if frame.get("method") == "turn/interrupt"
    ] == [
        {"threadId": _THREAD, "turnId": _TURN},
    ]


@pytest.mark.parametrize(
    ("acknowledge", "complete"), [(True, False), (False, False), (True, True), (False, True)]
)
async def test_eof_drains_notifications_before_classifying_the_turn(
    workspace: Path, acknowledge: bool, complete: bool
) -> None:
    steps = [
        *initialize(),
        *thread_setup(workspace),
        expect("turn/start", "run"),
        notify("turn/started", threadId=_THREAD, turn=turn()),
    ]
    if acknowledge:
        steps.append(reply("run", {"turn": turn()}))
    steps.append(
        notify(
            "item/agentMessage/delta",
            threadId=_THREAD,
            turnId=_TURN,
            itemId="text",
            delta="Visible",
        )
    )
    if complete:
        steps.append(notify("turn/completed", threadId=_THREAD, turn=turn(_TURN, "completed")))
    mock = mock_server(workspace, steps, exit_after_steps=True)
    async with (
        asyncio.timeout(8),
        CodexAppServerDriver().open(
            mock.profile, workspace_path=workspace, access="read_only"
        ) as session,
    ):
        result = await session.run_turn(
            TurnRequest(run_id="eof", prompt="Reply"), Events(), deny_all
        )
    assert result.result.status == ("completed" if complete else "indeterminate")
    assert result.result.text == "Visible"
    if not complete:
        assert result.result.reason == "codex_disconnected_before_turn_completed"
    mock.frames()


async def test_authentication_error_and_opt_in_raw_redaction(workspace: Path) -> None:
    error = {
        "message": "401 Unauthorized token=fixture-secret",
        "codexErrorInfo": {"httpConnectionFailed": {"httpStatusCode": 401}},
    }
    steps = [
        *initialize(),
        *thread_setup(workspace),
        expect("turn/start", "run"),
        reply("run", {"turn": turn()}),
        notify(
            "vendor/sample",
            threadId=_THREAD,
            turnId=_TURN,
            data={"apiKey": "fixture-secret", "safe": "hello"},
        ),
        notify("error", threadId=_THREAD, turnId=_TURN, error=error, willRetry=True),
        notify("turn/completed", threadId=_THREAD, turn=turn(_TURN, "failed", error=error)),
    ]
    mock = mock_server(workspace, steps, options={"include_raw_other": True})
    events = Events()
    async with CodexAppServerDriver().open(
        mock.profile, workspace_path=workspace, access="read_only"
    ) as session:
        result = await session.run_turn(
            TurnRequest(run_id="auth", prompt="Reply"), events, deny_all
        )
    assert result.result.status == "failed" and result.result.reason == "auth_required"
    (observed,) = [event for event in events if isinstance(event, ErrorEvent)]
    assert observed.code == "auth_required" and observed.retryable
    (extra,) = [event for event in events if isinstance(event, OtherEvent)]
    assert extra.redacted and extra.raw["data"] == {"apiKey": "<redacted>", "safe": "hello"}
    assert "fixture-secret" not in str(events)
    mock.frames()


@pytest.mark.parametrize("source", ["notification", "terminal"])
async def test_codex_uses_the_latest_error_after_a_retryable_401(
    workspace: Path, source: str
) -> None:
    params = {"threadId": _THREAD, "turnId": _TURN}
    retryable_error = {
        "message": "Unauthorized. Authentication failed.",
        "codexErrorInfo": {"httpConnectionFailed": {"httpStatusCode": 401}},
    }
    final_error = {"message": "The prompt is too long", "codexErrorInfo": "contextWindowExceeded"}
    terminal = turn(_TURN, "failed")
    steps = [
        *initialize(),
        *thread_setup(workspace),
        expect("turn/start", "run"),
        reply("run", {"turn": turn()}),
        notify("error", **params, error=retryable_error, willRetry=True),
    ]
    if source == "notification":
        steps.append(notify("error", **params, error=final_error, willRetry=False))
    else:
        terminal["error"] = final_error
    steps.append(notify("turn/completed", threadId=_THREAD, turn=terminal))
    mock = mock_server(workspace, steps)
    events = Events()
    async with CodexAppServerDriver().open(
        mock.profile, workspace_path=workspace, access="read_only"
    ) as session:
        result = await session.run_turn(
            TurnRequest(run_id="retry", prompt="Reply"), events, deny_all
        )
    assert (result.result.status, result.result.reason) == ("failed", "contextWindowExceeded")
    assert [(event.code, event.retryable) for event in events if isinstance(event, ErrorEvent)] == [
        ("auth_required", True),
        ("contextWindowExceeded", False),
    ]
    mock.frames()


async def test_replay_the_captured_blocked_transcript(workspace: Path) -> None:
    steps = []
    prompts = []
    for row in _TRANSCRIPT:
        frame = copy.deepcopy(row["message"])
        label = f"frame-{frame['id']}" if "id" in frame else None
        if row["direction"] == "client":
            params = frame.get("params", {})
            if frame["method"] == "initialize":
                params = {"clientInfo": {"name": "john", "version": "0.1.0"}}
            elif frame["method"] == "thread/start":
                params.update(cwd=str(workspace), approvalsReviewer="user")
            elif frame["method"] == "turn/start":
                params.pop("approvalPolicy", None)
                prompts.append(params["input"][0]["text"])
            elif frame["method"] == "turn/interrupt":
                steps.append({"signal": "interrupt"})
            steps.append(expect(frame["method"], label, **params))
        elif label is not None:
            if frame["id"] == 2:
                frame["result"]["cwd"] = str(workspace)
                restore_thread_fields(frame["result"]["thread"], workspace)
            del frame["id"]
            steps.append({"send": frame, "reply_to": label})
        else:
            if frame["method"] == "thread/started":
                restore_thread_fields(frame["params"]["thread"], workspace)
            if (
                frame["method"] == "thread/status/changed"
                and frame["params"]["status"]["type"] == "<redacted>"
            ):
                frame["params"]["status"] = {"type": "idle"}
            steps.append({"send": frame})

    mock = mock_server(workspace, steps, options={"approval_policy": "never"})
    async with (
        asyncio.timeout(8),
        CodexAppServerDriver().open(
            mock.profile, workspace_path=workspace, access="read_only"
        ) as session,
    ):
        assert session.capabilities.approvals == "none"
        errors = Events()
        first = await session.run_turn(
            TurnRequest(run_id="blocked", prompt=prompts[0]), errors, deny_all
        )
        assert first.result.status == "failed"
        assert any(isinstance(event, ErrorEvent) and event.retryable for event in errors)
        running = asyncio.create_task(
            session.run_turn(TurnRequest(run_id="interrupt", prompt=prompts[1]), Events(), deny_all)
        )
        await mock.wait_signal("interrupt")
        await session.cancel("interrupt")
        second = await running
    assert second.result.status == "cancelled"
    assert len(mock.frames()) == sum(row["direction"] == "client" for row in _TRANSCRIPT)
