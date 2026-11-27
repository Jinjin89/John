import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest

from john.drivers.claude import ClaudeStreamJsonDriver
from john.types import (
    AgentProfile,
    ApprovalDecision,
    ApprovalRequest,
    AuthRequiredError,
    BusyError,
    Capabilities,
    ErrorEvent,
    OtherEvent,
    RuntimeEvent,
    StartedEvent,
    TextEvent,
    ThoughtEvent,
    ToolEvent,
    TransportError,
    TurnRequest,
    UnsupportedError,
    UsageEvent,
    deny_all,
)

_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "claude"
_REPLAY = """
import json, sys
for line in open(sys.argv[-1], encoding='utf-8'):
    row = json.loads(line)
    frame = row['frame']
    if row['direction'] == 'incoming':
        if frame.get('type') == 'control_response':
            frame['response']['request_id'] = 'john-init-1'
        print(json.dumps(frame), flush=True)
    else:
        if frame.get('request', {}).get('subtype') == 'initialize':
            frame['request_id'] = 'john-init-1'
        got = json.loads(input())
        assert got == frame, (got, frame)
"""
_MOCK = """
import json, os, sys

mode, _, arguments = sys.argv[-1].partition(':')
flags = ['-p', '--input-format', 'stream-json', '--output-format', 'stream-json',
         '--verbose', '--include-partial-messages', '--permission-prompts', 'host',
         '--permission-prompt-tool', 'stdio', '--permission-mode']
assert sys.argv[1:len(flags) + 1] == flags
if mode == 'resume':
    assert sys.argv[-4:] == ['plan', '--resume', 'restored', 'resume']
else:
    assert sys.argv[-2] == 'manual'
if mode == 'child_env':
    assert 'ANTHROPIC_API_KEY' not in os.environ
    assert os.environ['ANTHROPIC_AUTH_TOKEN'] == 'profile-fixture'
    assert os.environ['JOHN_CLAUDE_MOCK'] == 'private-launch'

def send(type_, **fields):
    print(json.dumps({'type': type_, **fields}), flush=True)

def receive(type_):
    frame = json.loads(input())
    assert frame['type'] == type_, frame
    return frame

def control(rid, subtype, **fields):
    send('control_request', request_id=rid, request={'subtype': subtype, **fields})

def delta(kind, index, **fields):
    send('stream_event', event={'type': 'content_block_delta', 'index': index,
                                'delta': {'type': kind, **fields}})

def result(**fields):
    send('result', **{'subtype': 'success', 'is_error': False, 'session_id': session, **fields})

initialize = receive('control_request')
assert initialize == {'type': 'control_request', 'request_id': 'john-init-1',
                      'request': {'subtype': 'initialize'}}
if mode == 'timeout':
    sys.stdin.read()
    sys.exit(0)
send('control_response', response={'request_id': 'different-request', 'subtype': 'error'})
if mode == 'initialize_error':
    send('control_response', response={'request_id': 'john-init-1', 'subtype': 'error',
                                       **json.loads(arguments)})
    assert not sys.stdin.read()
    sys.exit(0)
effective = {'mode_mismatch': 'acceptEdits', 'resume': 'plan'}.get(mode, 'default')
send('control_response', response={'request_id': 'john-init-1', 'subtype': 'success',
                                   'response': {'current_permission_mode': effective}})
if mode in ('mode_mismatch', 'child_env'):
    assert not sys.stdin.read()
    sys.exit(0)
user = receive('user')
assert user['parent_tool_use_id'] is None and user['message']['role'] == 'user'
assert user['message']['content'] == [{'type': 'text', 'text': 'Run'}]
assert user['session_id'] == ('restored' if mode == 'resume' else '')
session = 'restored' if mode == 'resume' else 'session'
send('system', subtype='init', session_id=session)

if mode in ('error', 'auth'):
    error = 'Unauthorized' if mode == 'auth' else 'API Error: token=diagnostic-secret'
    send('assistant', is_api_error_message=True, session_id=session,
         message={'content': [{'type': 'text', 'text': error}]})
    result(is_error=True, result=error, api_error_status=401 if mode == 'auth' else 502)
    sys.exit(0)

if mode == 'auth_diagnostic':
    assistant, terminal = json.loads(arguments)
    if assistant is not None:
        send('assistant', session_id=session, **assistant)
    result(is_error=True, **terminal)
    sys.exit(0)

if mode == 'tool_auth_text':
    message = 'HTTP 401: Authentication failed'
    send('assistant', session_id=session, message={'content': [
        {'type': 'text', 'text': message},
        {'type': 'tool_use', 'id': 'tool-1', 'name': 'Read', 'input': {'error': message}},
    ]})
    send('user', session_id=session, message={'role': 'user', 'content': [
        {'type': 'tool_result', 'tool_use_id': 'tool-1', 'is_error': True, 'content': message},
    ]})
    result(is_error=True, result='The turn failed')
    sys.exit(0)

if mode == 'rich':
    assert os.getenv('JOHN_CLAUDE_MOCK') == 'private-launch'
    send('future_frame', api_key='plain-secret', safe_value=4)
    send('stream_event', event={'type': 'message_start', 'message': {'id': 'message-1'}})
    delta('thinking_delta', 0, thinking='Prepare')
    send('assistant', message={'id': 'message-1',
                               'content': [{'type': 'thinking', 'thinking': 'Prepare'}]})
    delta('text_delta', 1, text='A')
    send('assistant', message={'id': 'message-1', 'content': [{'type': 'text', 'text': 'AB'}]})
    control('allow', 'can_use_tool', tool_name='Read', input={'paths': ['file.txt']})
    control('unknown', 'future_operation')
    assert receive('control_response')['response'] == {
        'request_id': 'unknown', 'subtype': 'error', 'error': 'Unsupported control request',
    }
    delta('thinking_delta', 2, thinking='Approval pending')
    allow = receive('control_response')['response']
    assert allow == {'request_id': 'allow', 'subtype': 'success',
                     'response': {'behavior': 'allow', 'updatedInput': {'paths': ['file.txt']}}}
    control('deny', 'can_use_tool', tool_name='Write', input={})
    deny = receive('control_response')['response']
    assert deny == {'request_id': 'deny', 'subtype': 'success',
                    'response': {'behavior': 'deny', 'message': 'policy refused'}}
    result(usage={'input_tokens': 12, 'output_tokens': 5}, total_cost_usd=0.021,
           structured_output=None)
    sys.exit(0)

if mode == 'tool_results':
    send('assistant', message={'id': 'tools', 'content': [
        {'type': 'tool_use', 'id': 'read-1', 'name': 'Read'},
        {'type': 'tool_use', 'id': 'write-1', 'name': 'Write'},
    ]})
    outputs = [
        {'type': 'text', 'text': 'Tool results'},
        {'type': 'tool_result', 'tool_use_id': 'read-1', 'content': 'Read token=result-secret'},
        {'type': 'tool_result', 'tool_use_id': 'write-1', 'is_error': True,
         'content': [{'type': 'text', 'text': 'Denied password=write-secret'}]},
        {'type': 'tool_result', 'tool_use_id': 'unknown', 'content': 'Replay'},
    ]
    for _ in range(2):
        send('user', message={'role': 'user', 'content': outputs})
    result()
    sys.exit(0)

if mode == 'rewrite':
    send('stream_event', event={'type': 'message_start', 'message': {'id': 'rewrite'}})
    delta('text_delta', 0, text='draft')
    send('assistant', message={'id': 'rewrite',
                               'content': [{'type': 'text', 'text': 'revised'}]})
    result()
    sys.exit(0)

delta('text_delta', 0, text='partial')
if mode == 'eof':
    sys.exit(0)
if mode in ('cancel', 'deny_cancel'):
    control('pending', 'can_use_tool', tool_name='Read', input={})
    if mode == 'cancel':
        interrupt = receive('control_request')
        decision = receive('control_response')['response']
    else:
        decision = receive('control_response')['response']
        interrupt = receive('control_request')
    assert interrupt['request'] == {'subtype': 'interrupt'}
    assert decision['request_id'] == 'pending' and decision['response']['behavior'] == 'deny'
    send('control_response', response={'request_id': interrupt['request_id'], 'subtype': 'success'})
    result(is_error=True, terminal_reason='aborted_streaming')
else:
    result(result='restored answer')
"""


def _profile(tmp_path: Path, script: str, arg: str, **options: Any) -> AgentProfile:
    executable = tmp_path / "stream-mock"
    executable.write_text(f"#!{sys.executable}\n{script}", encoding="utf-8")
    executable.chmod(0o700)
    return AgentProfile(
        name="mock-stdio",
        kind="claude",
        command=str(executable),
        args=(arg,),
        env={"JOHN_CLAUDE_MOCK": "private-launch"},
        capabilities=Capabilities(enforced_sandbox=frozenset({"read_only"})),
        options=options,
    )


def _sink(records: list[RuntimeEvent]):
    async def emit(event: RuntimeEvent) -> None:
        records.append(event)

    return emit


async def test_claude_removes_configured_environment_variables_before_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "parent-key-fixture")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "parent-token-fixture")
    profile = _profile(tmp_path, _MOCK, "child_env")
    profile = profile.model_copy(
        update={
            "unset_env": ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"),
            "env": {**profile.env, "ANTHROPIC_AUTH_TOKEN": "profile-fixture"},
        }
    )
    async with ClaudeStreamJsonDriver().open(profile, workspace_path=tmp_path, access="read_only"):
        pass
    assert os.environ["ANTHROPIC_API_KEY"] == "parent-key-fixture"
    assert os.environ["ANTHROPIC_AUTH_TOKEN"] == "parent-token-fixture"


@pytest.mark.parametrize("fixture", ["cli.jsonl", "model-reference.jsonl"])
async def test_claude_golden_replay(tmp_path: Path, fixture: str) -> None:
    path = _FIXTURES / fixture
    frames = [json.loads(line)["frame"] for line in path.read_text().splitlines()]
    prompt = next(frame for frame in frames if frame.get("type") == "user")
    text = prompt["message"]["content"][0]["text"]
    complete = next(frame for frame in frames if frame.get("type") == "result")
    deny = next(
        frame
        for frame in frames
        if frame.get("type") == "control_response"
        and frame["response"].get("response", {}).get("behavior") == "deny"
    )
    calls: list[ApprovalRequest] = []
    events: list[RuntimeEvent] = []

    async def policy(request: ApprovalRequest) -> ApprovalDecision:
        calls.append(request)
        return ApprovalDecision(reason=deny["response"]["response"]["message"])

    async with (
        asyncio.timeout(8),
        ClaudeStreamJsonDriver().open(
            _profile(tmp_path, _REPLAY, str(path)), workspace_path=tmp_path, access="read_only"
        ) as session,
    ):
        outcome = await session.run_turn(
            TurnRequest(run_id="fixture", prompt=text), _sink(events), policy
        )
        assert session.native_ref == complete["session_id"]
    result = outcome.result
    assert result.status == "completed" and result.text == "OK"
    assert result.usage is not None and result.usage["total_cost_usd"] == complete["total_cost_usd"]
    assert (
        isinstance(events[0], StartedEvent)
        and events[0].native_session_ref == result.native_session_ref
    )
    assert len(calls) == 1 and calls[0].action == "Write"
    tools = [event for event in events if isinstance(event, ToolEvent)]
    assert (tools[0].name, tools[0].phase) == ("Write", "started")
    if fixture == "model-reference.jsonl":
        assert len(tools) == 1
        assert [event.text for event in events if isinstance(event, ThoughtEvent)] == [
            "Example thought"
        ]
        assert [event.text for event in events if isinstance(event, TextEvent)] == [
            "Trying Write.",
            "OK",
        ]
    else:
        assert len(tools) == 2
        failed = tools[1]
        assert (failed.call_id, failed.name, failed.phase) == (tools[0].call_id, "Write", "failed")
        assert failed.summary == "The protocol probe denies all tool calls."


async def test_claude_tool_result_events_are_correlated_redacted_and_deduplicated(
    tmp_path: Path,
) -> None:
    events: list[RuntimeEvent] = []
    async with (
        asyncio.timeout(8),
        ClaudeStreamJsonDriver().open(
            _profile(tmp_path, _MOCK, "tool_results"), workspace_path=tmp_path, access="read_only"
        ) as session,
    ):
        outcome = await session.run_turn(
            TurnRequest(run_id="tools", prompt="Run"), _sink(events), deny_all
        )
    assert outcome.result.status == "completed"
    assert [
        (event.call_id, event.name, event.phase, event.summary)
        for event in events
        if isinstance(event, ToolEvent)
    ] == [
        ("read-1", "Read", "started", ""),
        ("write-1", "Write", "started", ""),
        ("read-1", "Read", "completed", "Read token=<redacted>"),
        ("write-1", "Write", "failed", "Denied password=<redacted>"),
    ]


async def test_claude_divergent_snapshot_corrects_streamed_text(tmp_path: Path) -> None:
    events: list[RuntimeEvent] = []
    async with (
        asyncio.timeout(8),
        ClaudeStreamJsonDriver().open(
            _profile(tmp_path, _MOCK, "rewrite"), workspace_path=tmp_path, access="read_only"
        ) as session,
    ):
        outcome = await session.run_turn(
            TurnRequest(run_id="rewrite", prompt="Run"), _sink(events), deny_all
        )
    assert outcome.result.status == "completed" and outcome.result.text == "revised"
    assert [(event.text, event.is_delta) for event in events if isinstance(event, TextEvent)] == [
        ("draft", True),
        ("revised", False),
    ]


@pytest.mark.parametrize("include_raw", [False, True])
async def test_claude_partial_snapshot_and_parallel_controls(
    tmp_path: Path, include_raw: bool
) -> None:
    profile = _profile(tmp_path, _MOCK, "rich", include_raw_other=include_raw)
    events: list[RuntimeEvent] = []
    started = asyncio.Event()
    unknown_answered = asyncio.Event()
    release = asyncio.Event()
    approvals: list[ApprovalRequest] = []

    async def emit(event: RuntimeEvent) -> None:
        events.append(event)
        if isinstance(event, ThoughtEvent) and event.text == "Approval pending":
            unknown_answered.set()

    async def policy(request: ApprovalRequest) -> ApprovalDecision:
        approvals.append(request)
        if request.action == "Read":
            request.args["paths"].append("another.txt")
            started.set()
            await release.wait()
            return ApprovalDecision(outcome="allow_once")
        return ApprovalDecision(reason="policy refused")

    async with (
        asyncio.timeout(8),
        ClaudeStreamJsonDriver().open(
            profile, workspace_path=tmp_path, access="read_only"
        ) as session,
    ):
        active = asyncio.create_task(
            session.run_turn(TurnRequest(run_id="rich", prompt="Run"), emit, policy)
        )
        await started.wait()
        await unknown_answered.wait()
        with pytest.raises(BusyError):
            await session.run_turn(TurnRequest(run_id="overlap", prompt="Run"), _sink([]), deny_all)
        release.set()
        outcome = await active
    result = outcome.result
    assert result.status == "completed" and result.text == "AB"
    assert result.structured is None and result.has_structured
    assert result.usage == {"input_tokens": 12, "output_tokens": 5, "total_cost_usd": 0.021}
    assert [event.text for event in events if isinstance(event, TextEvent)] == ["A", "B"]
    assert [event.text for event in events if isinstance(event, ThoughtEvent)] == [
        "Prepare",
        "Approval pending",
    ]
    assert [event.cost_usd for event in events if isinstance(event, UsageEvent)] == [0.021]
    assert [item.action for item in approvals] == ["Read", "Write"]
    (unknown,) = [event for event in events if isinstance(event, OtherEvent)]
    assert unknown.native_type == "future_frame" and unknown.redacted
    assert unknown.raw == (
        {"type": "future_frame", "api_key": "<redacted>", "safe_value": 4} if include_raw else None
    )
    assert [event.seq for event in events] == list(range(len(events)))


@pytest.mark.parametrize(
    ("mode", "reason"), [("error", "API Error: token=<redacted>"), ("auth", "auth_required")]
)
async def test_claude_result_checks_is_error_on_success_subtype(
    tmp_path: Path, mode: str, reason: str
) -> None:
    events: list[RuntimeEvent] = []
    async with (
        asyncio.timeout(8),
        ClaudeStreamJsonDriver().open(
            _profile(tmp_path, _MOCK, mode), workspace_path=tmp_path, access="read_only"
        ) as session,
    ):
        outcome = await session.run_turn(
            TurnRequest(run_id=mode, prompt="Run"), _sink(events), deny_all
        )
    assert outcome.result.status == "failed" and outcome.result.reason == reason
    assert not outcome.result.text
    assert any(
        isinstance(event, ErrorEvent) and "diagnostic-secret" not in event.message
        for event in events
    )


@pytest.mark.parametrize(
    ("assistant", "terminal", "reason", "event_code"),
    [
        pytest.param(
            {
                "is_api_error_message": True,
                "message": {
                    "content": [
                        {
                            "type": "text",
                            "text": "Your account can't receive replies from Claude right now. "
                            "Visit portal.anthropic.com/verify.",
                        }
                    ]
                },
            },
            {"api_error_status": 403},
            "auth_required",
            "auth_required",
            id="account-verification",
        ),
        pytest.param(
            {"api_error_status": 401},
            {},
            "auth_required",
            "auth_required",
            id="assistant-401",
        ),
        pytest.param(
            None,
            {"api_error_status": 401, "result": "The provider rejected the request"},
            "auth_required",
            None,
            id="result-401",
        ),
        pytest.param(
            {"error": {"type": "invalid_request_error", "code": "invalid_api_key"}},
            {},
            "auth_required",
            "auth_required",
            id="structured-invalid-key",
        ),
        pytest.param(
            {"error": "authentication_failed"},
            {},
            "auth_required",
            "auth_required",
            id="sdk-authentication-failed",
        ),
        pytest.param(
            {"error": {"type": "permission_error", "message": "Unauthorized to use this model"}},
            {"api_error_status": 403, "result": "Unauthorized to use this model"},
            "Unauthorized to use this model",
            "api_error",
            id="permission-denial",
        ),
        pytest.param(
            None,
            {"api_error_status": 403, "result": "Finish signing in at the verification portal"},
            "auth_required",
            None,
            id="403-with-explicit-verification",
        ),
        pytest.param(
            None,
            {"result": "API error: HTTP 401"},
            "auth_required",
            None,
            id="text-only-401",
        ),
    ],
)
async def test_claude_classifies_native_auth_diagnostics(
    tmp_path: Path,
    assistant: dict[str, Any] | None,
    terminal: dict[str, Any],
    reason: str,
    event_code: str | None,
) -> None:
    profile = _profile(tmp_path, _MOCK, "auth_diagnostic:" + json.dumps([assistant, terminal]))
    events: list[RuntimeEvent] = []
    async with ClaudeStreamJsonDriver().open(
        profile, workspace_path=tmp_path, access="read_only"
    ) as session:
        result = await session.run_turn(
            TurnRequest(run_id="auth", prompt="Run"), _sink(events), deny_all
        )
    assert (result.result.status, result.result.reason) == ("failed", reason)
    assert [event.code for event in events if isinstance(event, ErrorEvent)] == (
        [event_code] if event_code else []
    )


@pytest.mark.parametrize(
    ("diagnostic", "exception"),
    [
        ({"api_error_status": 401, "error": "Provider rejected the request"}, AuthRequiredError),
        (
            {"error": {"type": "authentication_error", "message": "Login unavailable"}},
            AuthRequiredError,
        ),
        ({"api_error_status": 403, "error": "Unauthorized to use this model"}, TransportError),
    ],
)
async def test_claude_classifies_initialize_errors(
    tmp_path: Path, diagnostic: dict[str, Any], exception: type[Exception]
) -> None:
    profile = _profile(tmp_path, _MOCK, "initialize_error:" + json.dumps(diagnostic))
    with pytest.raises(exception):
        async with ClaudeStreamJsonDriver().open(
            profile, workspace_path=tmp_path, access="read_only"
        ):
            pytest.fail("Opened a session after an initialization error")


async def test_claude_does_not_classify_auth_text_in_model_or_tool_payloads(tmp_path: Path) -> None:
    profile = _profile(tmp_path, _MOCK, "tool_auth_text")
    events: list[RuntimeEvent] = []
    async with ClaudeStreamJsonDriver().open(
        profile, workspace_path=tmp_path, access="read_only"
    ) as session:
        result = await session.run_turn(
            TurnRequest(run_id="tool-error", prompt="Run"), _sink(events), deny_all
        )
    assert (result.result.status, result.result.reason) == ("failed", "The turn failed")
    assert not any(isinstance(event, ErrorEvent) for event in events)
    assert [event.phase for event in events if isinstance(event, ToolEvent)] == [
        "started",
        "failed",
    ]


@pytest.mark.parametrize("mode", ["cancel", "deny_cancel"])
async def test_claude_interrupt_drains_cancelled_result(tmp_path: Path, mode: str) -> None:
    pending = asyncio.Event()
    stopped = asyncio.Event()

    async def policy(request: ApprovalRequest) -> ApprovalDecision:
        pending.set()
        if mode == "deny_cancel":
            return ApprovalDecision(outcome="deny_cancel")
        try:
            await asyncio.Future()
        finally:
            stopped.set()
        return ApprovalDecision(outcome="allow_once")

    async with (
        asyncio.timeout(8),
        ClaudeStreamJsonDriver().open(
            _profile(tmp_path, _MOCK, mode), workspace_path=tmp_path, access="read_only"
        ) as session,
    ):
        active = asyncio.create_task(
            session.run_turn(TurnRequest(run_id=mode, prompt="Run"), _sink([]), policy)
        )
        await pending.wait()
        if mode == "cancel":
            await session.cancel(mode)
            await stopped.wait()
        reply = await active
    assert reply.result.status == "cancelled" and reply.result.text == "partial"


async def test_claude_unexpected_eof_is_indeterminate(tmp_path: Path) -> None:
    async with (
        asyncio.timeout(8),
        ClaudeStreamJsonDriver().open(
            _profile(tmp_path, _MOCK, "eof"), workspace_path=tmp_path, access="read_only"
        ) as session,
    ):
        result = await session.run_turn(
            TurnRequest(run_id="eof", prompt="Run"), _sink([]), deny_all
        )
    assert result.result.status == "indeterminate" and result.result.text == "partial"


async def test_claude_resume_and_permission_mode_flags(tmp_path: Path) -> None:
    profile = _profile(tmp_path, _MOCK, "resume", permission_mode="plan", unused=True)
    async with (
        asyncio.timeout(8),
        ClaudeStreamJsonDriver().open(
            profile, workspace_path=tmp_path, access="read_only", resume_ref="restored"
        ) as session,
    ):
        assert session.capabilities.resume and session.ignored_fields == {
            "unused": "unsupported option"
        }
        result = await session.run_turn(
            TurnRequest(run_id="resumed", prompt="Run"), _sink([]), deny_all
        )
    assert result.result.status == "completed" and result.result.native_session_ref == "restored"


async def test_claude_initialization_times_out(tmp_path: Path) -> None:
    profile = _profile(tmp_path, _MOCK, "timeout", init_timeout_s=0.1)
    with pytest.raises(TransportError, match="initialize timed out"):
        async with (
            asyncio.timeout(8),
            ClaudeStreamJsonDriver().open(profile, workspace_path=tmp_path, access="read_only"),
        ):
            pytest.fail("Opened a session without initialize confirmation")


async def test_claude_rejects_a_different_effective_permission_mode(tmp_path: Path) -> None:
    profile = _profile(tmp_path, _MOCK, "mode_mismatch")
    with pytest.raises(UnsupportedError, match="different permission mode"):
        async with (
            asyncio.timeout(8),
            ClaudeStreamJsonDriver().open(profile, workspace_path=tmp_path, access="read_only"),
        ):
            pytest.fail("Opened a session with a different permission mode")


async def test_claude_rejects_default_workspace_scope_before_launch(tmp_path: Path) -> None:
    profile = AgentProfile(name="unscoped", kind="claude", command=str(tmp_path / "missing"))
    with pytest.raises(UnsupportedError, match="requested workspace scope"):
        async with ClaudeStreamJsonDriver().open(
            profile, workspace_path=tmp_path, access="read_only"
        ):
            pytest.fail("Opened a workspace session with default capabilities")
