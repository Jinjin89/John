import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest

from john.drivers.acp import AcpDriver
from john.types import (
    AgentProfile,
    ApprovalDecision,
    ApprovalRequest,
    AuthRequiredError,
    BusyError,
    Capabilities,
    OtherEvent,
    OutputSpec,
    PlanEvent,
    ResumeUnavailableError,
    RuntimeEvent,
    StartedEvent,
    TextEvent,
    ThoughtEvent,
    ToolEvent,
    TransportError,
    TurnRequest,
    UnsupportedError,
    UsageEvent,
    WorkspaceRef,
    WorkspaceViolationError,
    deny_all,
)

_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "acp"
_REPLAY = """
import json, os, sys
for line in open(sys.argv[1], encoding='utf-8'):
    row = json.loads(line)
    frame = row['frame']
    if row['direction'] == 'incoming':
        print(json.dumps(frame), flush=True)
    else:
        if frame.get('method') == 'initialize':
            frame['params']['clientInfo']['name'] = 'john'
        elif frame.get('method') == 'session/new':
            frame['params']['cwd'] = os.getcwd()
        got = json.loads(input())
        assert got == frame, (got, frame)
"""
_MOCK = """
import json, os, sys, time
mode = sys.argv[1]

def send(**fields):
    print(json.dumps({'jsonrpc': '2.0', **fields}), flush=True)

def receive(method):
    frame = json.loads(input())
    assert frame['jsonrpc'] == '2.0' and frame['method'] == method, frame
    return frame

def update(kind, **fields):
    send(method='session/update', params={
        'sessionId': 'session', 'update': {'sessionUpdate': kind, **fields},
    })

def text(value):
    update('agent_message_chunk', content={'type': 'text', 'text': value})

def ask(rid, options):
    send(method='session/request_permission', id=rid, params={
        'sessionId': 'session', 'options': options,
        'toolCall': {'toolCallId': 'tool-1', 'name': 'Read', 'title': 'Read a file',
                     'rawInput': {'path': 'input.txt'}},
    })

init = receive('initialize')
assert init['params']['clientCapabilities'] == {'fs': {}, 'auth': {}}
assert init['params']['protocolVersion'] == 1
caps = {'loadSession': mode == 'load'}
if mode == 'resume':
    caps['sessionCapabilities'] = {'resume': {}}
send(id=init['id'], result={'protocolVersion': 2 if mode == 'version' else 1,
                           'agentCapabilities': caps})
if mode in ('version', 'unsupported'):
    assert not sys.stdin.read()
    sys.exit(0)

setup = receive({'load': 'session/load', 'resume': 'session/resume'}.get(mode, 'session/new'))
assert setup['params']['mcpServers'] == []
if mode == 'rich':
    assert setup['params']['cwd'] == '/agent/workspace'
    assert os.getcwd() == sys.argv[2] and os.getenv('JOHN_ACP_TEST_ENV') == 'launch'
else:
    assert setup['params']['cwd'] == os.getcwd()
if mode == 'auth':
    send(id=setup['id'], error={'code': -32000, 'message': 'Authentication required'})
    sys.exit(0)
if mode in ('load', 'resume'):
    assert setup['params']['sessionId'] == 'session'
    if mode == 'load':
        text('old replay')
        update('future_replay', secret='history')
    send(id=setup['id'], result={})
else:
    send(id=setup['id'], result={'sessionId': 'session'})
if mode == 'idle':
    assert not sys.stdin.read()
    sys.exit(0)

prompt = receive('session/prompt')
assert prompt['params']['sessionId'] == 'session'
if mode == 'rich':
    value = prompt['params']['prompt'][0]['text']
    assert '"path": "input.txt"' in value and '"path": "out.txt"' in value
    assert '"sha256": "' + 'a' * 64 in value
    update('agent_thought_chunk', content={'type': 'text', 'text': 'Consider the file'})
    update('tool_call', toolCallId='tool-1', title='Read a file', name='Read', status='pending')
    update('tool_call_update', toolCallId='tool-1', status='completed')
    update('plan', entries=[{'content': 'Read the input'}])
    update('plan_update', plan={'planId': 'p', 'type': 'markdown', 'content': 'Write the output'})
    update('plan_update', plan={'planId': 'p', 'type': 'items',
                                'entries': [{'content': 'Check the output'}]})
    update('usage_update', used=55, size=32768, cost={'amount': 0.02, 'currency': 'USD'})
    update('future_update', api_key='plain-secret', label='safe')
    send(method='future/notify', params={'token': 'test-token'})
    send(method='future/request', id='unhandled', params={})
    assert json.loads(input())['error']['code'] == -32601
    ask(7, [{'kind': 'allow_always', 'optionId': 'always'}])
    assert json.loads(input()) == {'jsonrpc': '2.0', 'id': 7,
                                   'result': {'outcome': {'outcome': 'cancelled'}}}
    ask('7', [{'kind': 'allow_always', 'optionId': 'always'},
              {'kind': 'reject_once', 'optionId': 'reject'}])
    assert json.loads(input()) == {'jsonrpc': '2.0', 'id': '7',
                                   'result': {'outcome': {'outcome': 'selected',
                                                          'optionId': 'reject'}}}
elif mode == 'deny_cancel':
    ask('decision', [{'kind': 'reject_once', 'optionId': 'reject'}])
    frames = [json.loads(input()), json.loads(input())]
    assert any(x.get('method') == 'session/cancel' for x in frames)
    reply = next(x for x in frames if x.get('id') == 'decision')
    assert reply['result']['outcome'] == {'outcome': 'selected', 'optionId': 'reject'}

text('live')
if mode == 'eof':
    sys.exit(0)
if mode == 'flood':
    for index in range(1100):
        text('backlogged')
        if index % 16 == 0:
            time.sleep(0.001)
text(' answer')
send(id=prompt['id'], result={
    'stopReason': {'deny_cancel': 'cancelled', 'max': 'max_tokens'}.get(mode, 'end_turn'),
    'usage': {'inputTokens': 30, 'outputTokens': 4},
})
"""


def _profile(script: str, *args: str, **options: Any) -> AgentProfile:
    return AgentProfile(
        name="test-rpc",
        kind="acp",
        command=sys.executable,
        args=("-u", "-c", script, *args),
        env={"JOHN_ACP_TEST_ENV": "launch"},
        capabilities=Capabilities(enforced_sandbox=frozenset({"read_only", "read_write"})),
        options=options,
    )


def _sink(records: list[RuntimeEvent]):
    async def emit(event: RuntimeEvent) -> None:
        records.append(event)

    return emit


@pytest.mark.parametrize("fixture", ["agy.jsonl", "mock.jsonl"])
async def test_acp_golden_replay(tmp_path: Path, fixture: str) -> None:
    profile = _profile(_REPLAY, str(_FIXTURES / fixture))
    events: list[RuntimeEvent] = []
    async with (
        asyncio.timeout(8),
        AcpDriver().open(profile, workspace_path=tmp_path, access="read_only") as session,
    ):
        if fixture == "agy.jsonl":
            assert session.capabilities.resume
            result = await session.run_turn(
                TurnRequest(run_id="agy", prompt="Reply with OK"), _sink(events), deny_all
            )
            assert result.result.status == "completed" and result.result.text == "OK"
            assert result.result.usage is not None and result.result.usage["inputTokens"] == 10170
            return

        async def allow(request: ApprovalRequest) -> ApprovalDecision:
            assert "allow_once" in request.offered_options
            return ApprovalDecision(outcome="allow_once")

        for name, policy in (("deny", deny_all), ("allow", allow)):
            reply = await session.run_turn(
                TurnRequest(run_id=name, prompt=name), _sink(events), policy
            )
            assert reply.result.status == "completed" and reply.result.text == "OK"
        assert any(isinstance(event, ThoughtEvent) for event in events)
        assert any(
            isinstance(event, OtherEvent)
            and event.native_type == "future_update"
            and event.raw is None
            and event.redacted
            for event in events
        )

        pending = asyncio.Event()
        released = asyncio.Event()

        async def blocked(request: ApprovalRequest) -> ApprovalDecision:
            pending.set()
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                released.set()
            return ApprovalDecision(outcome="allow_once")

        turn = asyncio.create_task(
            session.run_turn(TurnRequest(run_id="cancel", prompt="cancel"), _sink(events), blocked)
        )
        await pending.wait()
        with pytest.raises(BusyError):
            await session.run_turn(
                TurnRequest(run_id="overlap", prompt="extra"), _sink(events), deny_all
            )
        await session.cancel("different-run")
        assert not turn.done()
        await session.cancel("cancel")
        assert (await turn).result.status == "cancelled"
        await released.wait()
        assert any(
            isinstance(event, ToolEvent) and event.phase == "failed" and event.run_id == "cancel"
            for event in events
        )


async def test_acp_event_mapping_and_no_permission_escalation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("JOHN_ACP_TEST_ENV", "parent")
    profile = _profile(
        _MOCK,
        "rich",
        str(tmp_path),
        session_cwd="/agent/workspace",
        include_raw_other=True,
        unused_option=True,
    )
    request = TurnRequest(
        run_id="rich",
        prompt="Work with the input",
        input_refs=(
            WorkspaceRef(
                root="root",
                path="input.txt",
                media_type="text/plain",
                size_bytes=5,
                sha256="a" * 64,
            ),
        ),
        outputs=(OutputSpec(name="output", path="out.txt"),),
    )
    events: list[RuntimeEvent] = []
    approvals: list[ApprovalRequest] = []

    async def policy(req: ApprovalRequest) -> ApprovalDecision:
        approvals.append(req)
        return ApprovalDecision(outcome="allow_once")

    async with (
        asyncio.timeout(8),
        AcpDriver().open(profile, workspace_path=tmp_path, access="read_write") as session,
    ):
        reply = await session.run_turn(request, _sink(events), policy)

    assert reply.result.status == "completed" and reply.result.text == "live answer"
    assert reply.result.usage == {
        "used": 55,
        "size": 32768,
        "cost": {"amount": 0.02, "currency": "USD"},
        "inputTokens": 30,
        "outputTokens": 4,
    }
    assert [event.seq for event in events] == list(range(len(events)))
    assert isinstance(events[0], StartedEvent) and events[0].ignored_fields == {
        "unused_option": "unsupported option"
    }
    assert [event.items for event in events if isinstance(event, PlanEvent)] == [
        ("Read the input",),
        ("Write the output",),
        ("Check the output",),
    ]
    assert [(event.name, event.phase) for event in events if isinstance(event, ToolEvent)] == [
        ("Read", "started"),
        ("Read", "completed"),
    ]
    assert [event.cost_usd for event in events if isinstance(event, UsageEvent)] == [0.02, None]
    assert [event.text for event in events if isinstance(event, TextEvent)] == ["live", " answer"]
    other = {event.native_type: event.raw for event in events if isinstance(event, OtherEvent)}
    assert other["future_update"]["update"]["label"] == "safe"
    assert "plain-secret" not in json.dumps(other) and "test-token" not in json.dumps(other)
    assert [item.args for item in approvals] == [{"path": "input.txt"}] * 2
    assert len({item.request_id for item in approvals}) == 2
    assert os.environ["JOHN_ACP_TEST_ENV"] == "parent"


@pytest.mark.parametrize("mode", ["load", "resume"])
async def test_acp_restores_without_emitting_history(tmp_path: Path, mode: str) -> None:
    events: list[RuntimeEvent] = []
    async with (
        asyncio.timeout(8),
        AcpDriver().open(
            _profile(_MOCK, mode), workspace_path=tmp_path, access="read_only", resume_ref="session"
        ) as session,
    ):
        assert session.capabilities.resume and session.native_ref == "session"
        result = await session.run_turn(
            TurnRequest(run_id="resume", prompt="Continue"), _sink(events), deny_all
        )
    assert result.result.text == "live answer"
    assert [event.text for event in events if isinstance(event, TextEvent)] == ["live", " answer"]
    assert not any(isinstance(event, OtherEvent) for event in events)


@pytest.mark.parametrize(
    ("mode", "status"),
    [("eof", "indeterminate"), ("max", "incomplete"), ("deny_cancel", "cancelled")],
)
async def test_acp_terminal_classification(tmp_path: Path, mode: str, status: str) -> None:
    async def deny_and_cancel(request: ApprovalRequest) -> ApprovalDecision:
        return ApprovalDecision(outcome="deny_cancel")

    async with (
        asyncio.timeout(8),
        AcpDriver().open(
            _profile(_MOCK, mode), workspace_path=tmp_path, access="read_only"
        ) as session,
    ):
        result = await session.run_turn(
            TurnRequest(run_id=mode, prompt="Run"), _sink([]), deny_and_cancel
        )
    assert result.result.status == status
    assert result.result.text.startswith("live")


@pytest.mark.parametrize(
    ("mode", "error"),
    [
        ("unsupported", ResumeUnavailableError),
        ("version", UnsupportedError),
        ("auth", AuthRequiredError),
    ],
)
async def test_acp_rejects_unavailable_session(
    tmp_path: Path, mode: str, error: type[Exception]
) -> None:
    with pytest.raises(error):
        async with (
            asyncio.timeout(8),
            AcpDriver().open(
                _profile(_MOCK, mode),
                workspace_path=tmp_path,
                access="read_only",
                resume_ref="session" if mode == "unsupported" else None,
            ),
        ):
            pytest.fail("An invalid handshake opened a session")


async def test_acp_rejects_missing_command_and_unsafe_workspace(tmp_path: Path) -> None:
    driver = AcpDriver()
    with pytest.raises(TransportError, match="explicit command"):
        async with driver.open(
            AgentProfile(name="missing", kind="acp"), workspace_path=None, access="read_only"
        ):
            pass
    async with (
        asyncio.timeout(8),
        driver.open(
            _profile(_MOCK, "idle"), workspace_path=tmp_path, access="read_write"
        ) as session,
    ):
        request = TurnRequest(
            run_id="unsafe",
            prompt="hello",
            outputs=(OutputSpec(name="outside", path="../outside"),),
        )
        with pytest.raises(WorkspaceViolationError, match="relative"):
            await session.run_turn(request, _sink([]), deny_all)


async def test_acp_rejects_default_workspace_scope_before_launch(tmp_path: Path) -> None:
    profile = AgentProfile(name="unscoped", kind="acp", command=str(tmp_path / "missing"))
    with pytest.raises(UnsupportedError, match="requested workspace scope"):
        async with AcpDriver().open(profile, workspace_path=tmp_path, access="read_only"):
            pytest.fail("Opened a workspace session with default capabilities")


async def test_acp_turn_reports_overflow_when_event_sink_lags(tmp_path: Path) -> None:
    writing = asyncio.Event()
    release = asyncio.Event()

    async def emit(event: RuntimeEvent) -> None:
        if isinstance(event, TextEvent):
            writing.set()
            await release.wait()

    async with (
        asyncio.timeout(8),
        AcpDriver().open(
            _profile(_MOCK, "flood"), workspace_path=tmp_path, access="read_only"
        ) as session,
    ):
        turn = asyncio.create_task(
            session.run_turn(TurnRequest(run_id="flood", prompt="Run"), emit, deny_all)
        )
        try:
            await writing.wait()
            while session._peer.transport.error is None:
                await asyncio.sleep(0.01)
            release.set()
            outcome = await turn
        finally:
            turn.cancel()
            await asyncio.gather(turn, return_exceptions=True)
    assert outcome.result.status == "indeterminate"
    assert outcome.result.reason == "Transport frame queue overflow"
    assert outcome.result.text == "live"
