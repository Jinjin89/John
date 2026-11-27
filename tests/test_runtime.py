import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path

import pytest

from john.ledger import SqliteLedger
from john.runtime import ApprovalBroker, SessionRuntime, make_seq_allocator, validate_json_output
from john.types import (
    ApprovalDecision,
    ApprovalEvent,
    ApprovalPolicy,
    ApprovalRequest,
    BusyError,
    Capabilities,
    EventSink,
    FinishedEvent,
    RuntimeEvent,
    StartedEvent,
    TextEvent,
    TransportError,
    TurnFinished,
    TurnNeedsInput,
    TurnOutcome,
    TurnRequest,
    TurnResult,
)

type Program = Callable[[TurnRequest, EventSink, ApprovalPolicy], Awaitable[TurnOutcome]]


def approval(request_id: str = "permission") -> ApprovalRequest:
    return ApprovalRequest(request_id=request_id, run_id="run", action="read the sample")


async def test_default_policy_denies_and_records_both_phases() -> None:
    events: list[RuntimeEvent] = []

    async def emit(event: RuntimeEvent) -> None:
        events.append(event)

    broker = ApprovalBroker(emit_event=emit)
    decision = await broker.decide(approval())
    assert decision.outcome == "deny"
    assert await broker.decide(approval()) == decision
    assert all(isinstance(event, ApprovalEvent) for event in events)
    assert [(event.seq, event.phase, event.outcome) for event in events] == [
        (0, "requested", None),
        (1, "decided", "deny"),
    ]


async def test_parallel_approval_duplicates_share_one_policy_call() -> None:
    events = []
    policy_started, release = asyncio.Event(), asyncio.Event()
    calls = []

    async def emit(event: RuntimeEvent) -> None:
        events.append(event)

    async def policy(req: ApprovalRequest) -> ApprovalDecision:
        calls.append(req)
        assert [event.phase for event in events] == ["requested"]
        policy_started.set()
        await release.wait()
        return ApprovalDecision(outcome="allow_once", reason="authorized")

    broker = ApprovalBroker(policy, emit_event=emit)
    first = asyncio.create_task(broker.decide(approval()))
    async with asyncio.timeout(2):
        await policy_started.wait()
        duplicate = asyncio.create_task(broker.decide(approval()))
        release.set()
        first_result, duplicate_result = await asyncio.gather(first, duplicate)
    assert (
        first_result
        == duplicate_result
        == ApprovalDecision(outcome="allow_once", reason="authorized")
    )
    assert calls == [approval()]
    assert [event.phase for event in events] == ["requested", "decided"]
    broker.mark_cancelled()
    assert (await broker.decide(approval())).outcome == "cancelled"
    assert len(events) == 2


async def test_reused_approval_id_denies_changed_action_or_arguments() -> None:
    started, release = asyncio.Event(), asyncio.Event()
    calls = []

    async def policy(req: ApprovalRequest) -> ApprovalDecision:
        calls.append(req)
        started.set()
        await release.wait()
        return ApprovalDecision(outcome="allow_once")

    broker = ApprovalBroker(policy)
    original = approval().model_copy(update={"args": {"sample": "a", "count": 2}})
    first = asyncio.create_task(broker.decide(original))
    conflict = ApprovalDecision(reason="conflicting_approval_request_id")
    try:
        async with asyncio.timeout(2):
            await started.wait()
            assert await broker.decide(original.model_copy(update={"action": "other"})) == conflict
            release.set()
            assert (await first).outcome == "allow_once"
            assert (
                await broker.decide(original.model_copy(update={"args": {"count": 3}})) == conflict
            )
            reordered = original.model_copy(update={"args": {"count": 2, "sample": "a"}})
            assert (await broker.decide(reordered)).outcome == "allow_once"
    finally:
        release.set()
        await first
    assert calls == [original]


@pytest.mark.parametrize("cancel", [False, True], ids=["timeout", "cancel"])
async def test_approval_disregards_a_policy_that_returns_after_cancellation(cancel: bool) -> None:
    started, interrupted, release, finished = (
        asyncio.Event(),
        asyncio.Event(),
        asyncio.Event(),
        asyncio.Event(),
    )

    async def slow_policy(req: ApprovalRequest) -> ApprovalDecision:
        started.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            interrupted.set()
            await release.wait()
        finally:
            finished.set()
        return ApprovalDecision(outcome="allow_once")

    events = []

    async def emit(event: RuntimeEvent) -> None:
        events.append(event)

    broker = ApprovalBroker(slow_policy, timeout_s=5 if cancel else 0.05, emit_event=emit)
    resolving = asyncio.create_task(broker.decide(approval()))
    try:
        async with asyncio.timeout(2):
            await started.wait()
            if cancel:
                broker.mark_cancelled()
            decision = await resolving
            await interrupted.wait()
        expected = ("cancelled", "turn_cancelled") if cancel else ("deny", "approval_timeout")
        assert (decision.outcome, decision.reason) == expected
        assert not finished.is_set()
        assert events[-1].phase == "decided" and events[-1].outcome == decision.outcome
    finally:
        release.set()
        await asyncio.wait_for(finished.wait(), 2)
    assert await broker.decide(approval()) == decision
    assert len(events) == 2
    if cancel:
        assert await broker.decide(approval("after-cancel")) == decision


async def test_policy_failure_denies_without_leaking_exception_text() -> None:
    async def policy(req: ApprovalRequest) -> ApprovalDecision:
        raise RuntimeError("token=private")

    result = await ApprovalBroker(policy).decide(approval())
    assert (result.outcome, result.reason) == ("deny", "policy_error: RuntimeError")


def test_sequence_allocators_are_independent() -> None:
    first = make_seq_allocator()
    second = make_seq_allocator()
    assert [first(), first(), second(), first(), second()] == [0, 1, 0, 2, 1]


@pytest.mark.parametrize(
    ("value", "schema", "expected"),
    [
        ('{"amount": 12}', {"type": "object"}, {"amount": 12}),
        ("null", {"type": "null"}, None),
        ("false", {"type": "boolean"}, False),
    ],
)
def test_output_validation_accepts_json_values(value, schema, expected) -> None:
    validated = validate_json_output(
        TurnResult(run_id="run", status="completed", text=value), schema
    )
    assert validated.status == "completed"
    assert validated.has_structured is True and validated.structured == expected


def test_native_structured_result_takes_precedence_including_null() -> None:
    result = TurnResult(run_id="run", status="completed", text="summary", has_structured=True)
    validated = validate_json_output(result, {"type": "null"})
    assert (
        validated.status == "completed"
        and validated.has_structured
        and validated.structured is None
    )
    failed = TurnResult(run_id="run", status="indeterminate", reason="unknown")
    assert validate_json_output(failed, {"type": "object"}) is failed


@pytest.mark.parametrize(
    ("value", "schema", "reason"),
    [
        ("not JSON", {}, "invalid_output"),
        ("NaN", {"type": "number"}, "invalid_output"),
        ("1e999", {"type": "number"}, "invalid_output"),
        ('"secret"', {"type": "integer"}, "invalid_output: schema constraint 'type' failed"),
        ("null", {"type": "invalid"}, "invalid_schema"),
    ],
)
def test_output_validation_rejects_invalid_values(value, schema, reason) -> None:
    result = TurnResult(run_id="run", status="completed", text=value)
    validated = validate_json_output(result, schema)
    assert (validated.status, validated.reason) == ("failed", reason)


class TestSession:
    __test__ = False
    native_ref = "session"
    capabilities = Capabilities()
    ignored_fields: dict[str, str] = {}

    def __init__(self, program: Program):
        self.program = program
        self.stop = asyncio.Event()
        self.cancel_calls: list[str] = []

    async def run_turn(
        self, request: TurnRequest, on_event: EventSink, on_approval: ApprovalPolicy
    ) -> TurnOutcome:
        return await self.program(request, on_event, on_approval)

    async def cancel(self, run_id: str) -> None:
        self.cancel_calls.append(run_id)
        self.stop.set()


@pytest.fixture
async def runtime(tmp_path: Path) -> AsyncIterator[SessionRuntime]:
    ledger = await SqliteLedger.open(tmp_path / "ledger.db")
    try:
        yield SessionRuntime(ledger, cancel_timeout_s=0.05)
    finally:
        await ledger.close()


def turn_request(run_id: str = "run", **options) -> TurnRequest:
    return TurnRequest(run_id=run_id, prompt="Review the sample", **options)


async def execute(
    runtime: SessionRuntime,
    session: TestSession,
    request: TurnRequest | None = None,
    *,
    task_id: str = "task",
    **options,
) -> tuple[TurnOutcome, int]:
    return await runtime.execute_turn(
        session,
        request or turn_request(),
        owner="owner",
        agent_name="review",
        task_id=task_id,
        context_id="context",
        **options,
    )


async def test_observer_cannot_block_an_approval_or_the_ledger(runtime: SessionRuntime) -> None:
    observing, release, drained = asyncio.Event(), asyncio.Event(), asyncio.Event()
    seen = []

    async def observer(event: RuntimeEvent) -> None:
        observing.set()
        await release.wait()
        seen.append(event)
        if len(seen) == 4:
            drained.set()

    async def policy(req: ApprovalRequest) -> ApprovalDecision:
        journal = await runtime.ledger.get_run_events(req.run_id)
        assert journal[-1].phase == "requested"
        return ApprovalDecision(outcome="allow_once")

    async def program(request: TurnRequest, emit: EventSink, decide: ApprovalPolicy) -> TurnOutcome:
        run = await runtime.ledger.get_run(request.run_id)
        assert run is not None and run["status"] == "dispatched"
        await emit(StartedEvent(run_id=request.run_id, seq=99))
        await observing.wait()
        assert (await decide(approval())).outcome == "allow_once"
        await emit(TextEvent(run_id=request.run_id, seq=99, text="done"))
        await emit(FinishedEvent(run_id=request.run_id, seq=100, status="completed"))
        return TurnFinished(
            result=TurnResult(run_id=request.run_id, status="completed", text="null")
        )

    try:
        async with asyncio.timeout(2):
            result, next_seq = await execute(
                runtime,
                TestSession(program),
                turn_request(schema_def={"type": "null"}),
                on_event=observer,
                policy=policy,
            )
        assert isinstance(result, TurnFinished) and result.result.has_structured
        assert result.result.structured is None and result.result.status == "completed"
        assert next_seq == 4 and seen == []
        events = await runtime.ledger.get_run_events("run")
        assert [event.seq for event in events] == list(range(next_seq))
        assert [event.kind for event in events] == ["started", "approval", "approval", "text"]
    finally:
        release.set()
    await asyncio.wait_for(drained.wait(), 2)
    assert seen == events


async def test_overflow_disconnects_optional_observer_without_losing_events(
    runtime: SessionRuntime, caplog: pytest.LogCaptureFixture
) -> None:
    observing, disconnected = asyncio.Event(), asyncio.Event()

    async def observer(event: RuntimeEvent) -> None:
        observing.set()
        try:
            await asyncio.Future()
        finally:
            disconnected.set()

    async def policy(req: ApprovalRequest) -> ApprovalDecision:
        return ApprovalDecision(outcome="allow_once")

    async def program(turn: TurnRequest, emit: EventSink, decide: ApprovalPolicy) -> TurnOutcome:
        await emit(StartedEvent(run_id=turn.run_id, seq=0))
        await observing.wait()
        for index in range(258):
            await emit(TextEvent(run_id=turn.run_id, seq=index + 1, text="sample"))
        assert (await decide(approval())).outcome == "allow_once"
        return TurnFinished(result=TurnResult(run_id=turn.run_id, status="completed"))

    async with asyncio.timeout(15):
        outcome, next_seq = await execute(
            runtime, TestSession(program), on_event=observer, policy=policy
        )
        await disconnected.wait()
    assert isinstance(outcome, TurnFinished) and outcome.result.status == "completed"
    assert next_seq == 261
    journal = await runtime.ledger.get_run_events("run")
    assert [event.seq for event in journal] == list(range(next_seq))
    assert journal[-1].outcome == "allow_once"
    assert caplog.text.count("Runtime event observer disconnected") == 1


@pytest.mark.parametrize("value", [9_007_199_254_740_993, -9_007_199_254_740_993])
def test_nested_integer_must_fit_protobuf_number_precision(value: int) -> None:
    output = {"rows": [{"sample": value, "flagged": True}]}
    result = TurnResult(run_id="run", status="completed", structured=output, has_structured=True)
    validated = validate_json_output(result, {})
    assert validated.status == "failed"
    assert (
        validated.reason == "invalid_output: integer exceeds JSON/protobuf exact number precision"
    )
    output = {"rows": [{"sample": (1 << 53) - 1, "flagged": True}]}
    safe = validate_json_output(result.model_copy(update={"structured": output}), {})
    assert safe.status == "completed" and safe.structured == output


async def test_runtime_rejects_lossy_structured_result_without_schema(
    runtime: SessionRuntime,
) -> None:
    async def program(turn: TurnRequest, emit: EventSink, decide: ApprovalPolicy) -> TurnOutcome:
        return TurnFinished(
            result=TurnResult(
                run_id=turn.run_id, status="completed", structured=[1 << 53], has_structured=True
            )
        )

    outcome, _ = await execute(runtime, TestSession(program))
    assert isinstance(outcome, TurnFinished) and outcome.result.status == "failed"
    assert "exact number precision" in outcome.result.reason


async def test_runtime_rejects_overlap_and_returns_confirmed_cancel(
    runtime: SessionRuntime,
) -> None:
    started = asyncio.Event()

    async def program(request, emit, decide):
        started.set()
        await session.stop.wait()
        return TurnFinished(result=TurnResult(run_id=request.run_id, status="cancelled"))

    session = TestSession(program)
    running = asyncio.create_task(execute(runtime, session))
    async with asyncio.timeout(2):
        await started.wait()
        with pytest.raises(BusyError):
            await execute(runtime, session, turn_request("other"), task_id="other")
        with pytest.raises(BusyError):
            await execute(runtime, TestSession(program), turn_request("other"), task_id="other")
        cancel_result = await runtime.cancel_task("task")
        result = await running
    assert await runtime.ledger.get_run("other") is None
    assert cancel_result == result and session.cancel_calls == ["run"]
    assert isinstance(result[0], TurnFinished) and result[0].result.status == "cancelled"


async def test_runtime_deadline_invokes_native_cancel_and_drains(runtime: SessionRuntime) -> None:
    drain_runtime = SessionRuntime(runtime.ledger, cancel_timeout_s=1.0)

    async def program(request, emit, decide):
        await session.stop.wait()
        await emit(TextEvent(run_id=request.run_id, seq=9, text="stopped"))
        return TurnFinished(result=TurnResult(run_id=request.run_id, status="cancelled"))

    session = TestSession(program)
    async with asyncio.timeout(3):
        outcome, next_seq = await execute(drain_runtime, session, turn_request(timeout_s=0.02))
    assert isinstance(outcome, TurnFinished) and outcome.result.status == "cancelled"
    assert session.cancel_calls == ["run"] and next_seq == 1
    assert len(await runtime.ledger.get_run_events("run")) == 1


async def test_caller_cancellation_during_cleanup_releases_session(
    runtime: SessionRuntime, monkeypatch: pytest.MonkeyPatch
) -> None:
    finalizing, release = asyncio.Event(), asyncio.Event()
    original_close = ApprovalBroker._close

    async def held_close(broker: ApprovalBroker) -> None:
        finalizing.set()
        await release.wait()
        await original_close(broker)

    async def program(request, emit, decide):
        return TurnFinished(result=TurnResult(run_id=request.run_id, status="completed"))

    monkeypatch.setattr(ApprovalBroker, "_close", held_close)
    session = TestSession(program)
    caller = asyncio.create_task(execute(runtime, session))
    async with asyncio.timeout(2):
        try:
            await finalizing.wait()
            caller.cancel()
            with pytest.raises(asyncio.CancelledError):
                await caller
            drain = asyncio.create_task(runtime.cancel_task("task"))
            await asyncio.sleep(0)
        finally:
            release.set()
        result = await drain
        assert result is not None
        outcome, _ = result
        assert isinstance(outcome, TurnFinished) and outcome.result.status == "completed"
        assert session.cancel_calls == []
        subsequent, _ = await execute(runtime, session, turn_request("next"))
        assert isinstance(subsequent, TurnFinished) and subsequent.result.status == "completed"


async def test_late_driver_completion_cannot_replace_a_cancel_timeout(
    runtime: SessionRuntime,
) -> None:
    started, interrupted, release, exited = (
        asyncio.Event(),
        asyncio.Event(),
        asyncio.Event(),
        asyncio.Event(),
    )

    async def program(request, emit, decide):
        started.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            interrupted.set()
            await release.wait()
            await emit(TextEvent(run_id=request.run_id, seq=0, text="late"))
        finally:
            exited.set()
        return TurnFinished(result=TurnResult(run_id=request.run_id, status="completed"))

    session = TestSession(program)
    running = asyncio.create_task(execute(runtime, session))
    try:
        async with asyncio.timeout(2):
            await started.wait()
            cancelled = await runtime.cancel_task("task", timeout_s=0.02)
            attempt = await running
            await interrupted.wait()
        assert cancelled == attempt
        outcome, next_seq = attempt
        assert isinstance(outcome, TurnFinished)
        assert (outcome.result.status, outcome.result.reason, next_seq) == (
            "indeterminate",
            "cancel_timeout",
            0,
        )
        with pytest.raises(BusyError):
            await execute(runtime, session, turn_request("other"))
    finally:
        release.set()
        await asyncio.wait_for(exited.wait(), 2)
    assert await runtime.ledger.get_run_events("run") == []


@pytest.mark.parametrize(
    ("failure", "status", "reason"),
    [
        (None, "failed", "unsupported_task_question"),
        (TransportError("token=private"), "indeterminate", "transport_error"),
        (RuntimeError("private"), "indeterminate", "driver_error: RuntimeError"),
    ],
)
async def test_runtime_reports_uncertain_or_unsupported_outcomes(
    runtime: SessionRuntime, failure: Exception | None, status: str, reason: str
) -> None:
    async def program(request, emit, decide):
        if failure is not None:
            raise failure
        return TurnNeedsInput(run_id=request.run_id, question="Which month?")

    outcome, next_seq = await execute(runtime, TestSession(program))
    assert isinstance(outcome, TurnFinished)
    assert (outcome.result.status, outcome.result.reason, next_seq) == (status, reason, 0)
    assert (await runtime.ledger.get_run("run"))["status"] == "dispatched"
