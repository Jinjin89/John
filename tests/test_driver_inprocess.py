import asyncio
import hashlib
import json
from pathlib import Path

import pytest

from john.drivers import default_drivers
from john.drivers.inprocess import (
    InProcessDriver,
    SpecialistContext,
    make_cutoff_qc_specialist,
    make_echo_specialist,
)
from john.types import (
    AgentProfile,
    ApprovalDecision,
    ApprovalEvent,
    ApprovalRequest,
    BusyError,
    Capabilities,
    Driver,
    DriverSession,
    ErrorEvent,
    FinishedEvent,
    OutputSpec,
    ResumeUnavailableError,
    RuntimeEvent,
    TextEvent,
    ToolEvent,
    TurnFinished,
    TurnNeedsInput,
    TurnOutcome,
    TurnRequest,
    TurnResult,
    UnsupportedError,
    UsageEvent,
    WorkspaceRef,
    deny_all,
)

_CSV = "invoice_id,amount,expected_amount\npartial,50,100\nmatched,40,40\nmissing,0,20\n"
_JSON = json.dumps(
    {
        "items": [
            {"id": "partial", "score": 0.5},
            {"id": "matched", "score": 0},
            {"id": "missing", "score": 1},
        ]
    }
)


class Events(list[RuntimeEvent]):
    async def __call__(self, event: RuntimeEvent) -> None:
        self.append(event)


def specialist_profile(name: str, **options: object) -> AgentProfile:
    return AgentProfile(
        name=name,
        kind="inprocess",
        capabilities=Capabilities(enforced_sandbox={"read_only"}),
        options={"handler": name, **options},
    )


def csv_ref(path: str, data: bytes) -> WorkspaceRef:
    return WorkspaceRef(
        root="invoices",
        path=path,
        size_bytes=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
        media_type="text/csv",
    )


@pytest.mark.parametrize("source", ["inline-json", "inline-csv", "workspace-csv"])
async def test_cutoff_specialist_resumes_from_a_serialized_checkpoint(
    tmp_path: Path, source: str
) -> None:
    profile = specialist_profile("qc", handler="invoice-qc")
    driver = InProcessDriver({"invoice-qc": make_cutoff_qc_specialist()})
    assert isinstance(driver, Driver)
    prompt = _JSON if source == "inline-json" else _CSV
    data = prompt.encode()
    refs = ()
    workspace = None
    if source == "workspace-csv":
        workspace = tmp_path
        (tmp_path / "out").mkdir()
        (tmp_path / "out/invoices.csv").write_bytes(data)
        refs = (csv_ref("out/invoices.csv", data),)
        prompt = "Apply invoice quality control to the CSV artifact."

    start_events = Events()
    async with driver.open(profile, workspace_path=workspace, access="read_only") as session:
        assert isinstance(session, DriverSession)
        assert session.capabilities.task_questions
        paused = await session.run_turn(
            TurnRequest(run_id="score", prompt=prompt, input_refs=refs), start_events, deny_all
        )
    assert isinstance(paused, TurnNeedsInput)
    assert paused.run_id == "score"
    assert paused.question == "Which cutoff threshold should be applied?"
    checkpoint = json.loads(json.dumps(paused.continuation, allow_nan=False))
    assert checkpoint == {
        "stage": "awaiting_cutoff",
        "items": [
            {"invoice_id": "partial", "score": 0.5},
            {"invoice_id": "matched", "score": 0.0},
            {"invoice_id": "missing", "score": 1.0},
        ],
        "input_sha256": hashlib.sha256(data).hexdigest(),
    }
    assert [event.kind for event in start_events] == ["started", "thought", "tool", "tool"]
    assert [event.phase for event in start_events if isinstance(event, ToolEvent)] == [
        "started",
        "completed",
    ]
    assert [event.seq for event in start_events] == [0, 1, 2, 3]

    completion_events = Events()
    # Reopen with a fresh handler and pass only the restored checkpoint.
    fresh_driver = InProcessDriver({"invoice-qc": make_cutoff_qc_specialist()})
    async with fresh_driver.open(profile, workspace_path=workspace, access="read_only") as session:
        outcome = await session.run_turn(
            TurnRequest(run_id="apply", prompt="0.50", continuation=checkpoint),
            completion_events,
            deny_all,
        )
    assert isinstance(outcome, TurnFinished)
    assert outcome.result.status == "completed" and outcome.result.has_structured
    assert outcome.result.structured == {
        "cutoff": 0.5,
        "flagged": ["partial", "missing"],
        "total": 3,
        "passed": 1,
    }
    assert [event.kind for event in completion_events] == ["started", "text", "finished"]
    assert [event.seq for event in completion_events] == [0, 1, 2]
    assert {event.run_id for event in completion_events} == {"apply"}


async def test_invalid_cutoff_keeps_checkpoint_for_a_valid_retry() -> None:
    profile = specialist_profile("qc")
    driver = InProcessDriver({"qc": make_cutoff_qc_specialist()})
    async with driver.open(profile, workspace_path=None, access="read_only") as session:
        paused = await session.run_turn(
            TurnRequest(run_id="start", prompt=_CSV), Events(), deny_all
        )
        assert isinstance(paused, TurnNeedsInput)
        checkpoint = json.loads(json.dumps(paused.continuation))
        for index, answer in enumerate(("", "unknown", "-0.1", "1.1", "NaN", "inf")):
            retry = await session.run_turn(
                TurnRequest(run_id=f"retry-{index}", prompt=answer, continuation=checkpoint),
                Events(),
                deny_all,
            )
            assert isinstance(retry, TurnNeedsInput)
            assert retry.continuation == checkpoint
            assert "between 0 and 1" in retry.question
        resolved = await session.run_turn(
            TurnRequest(run_id="resolved", prompt="1", continuation=checkpoint), Events(), deny_all
        )
    assert isinstance(resolved, TurnFinished)
    assert resolved.result.structured["flagged"] == ["missing"]


async def test_multiple_workspace_refs_keep_row_order_and_fingerprint(tmp_path: Path) -> None:
    profile = specialist_profile("qc")
    driver = InProcessDriver({"qc": make_cutoff_qc_specialist()})
    refs = []
    for name, score in (("one", "0.4"), ("two", "0.9")):
        data = f"id,score\n{name},{score}\n".encode()
        path = f"{name}.csv"
        (tmp_path / path).write_bytes(data)
        refs.append(csv_ref(path, data))
    async with driver.open(profile, workspace_path=tmp_path, access="read_only") as session:
        outcome = await session.run_turn(
            TurnRequest(run_id="combine", prompt="Check both", input_refs=tuple(refs)),
            Events(),
            deny_all,
        )
    assert isinstance(outcome, TurnNeedsInput)
    assert [item["invoice_id"] for item in outcome.continuation["items"]] == ["one", "two"]
    assert (
        outcome.continuation["input_sha256"]
        == hashlib.sha256(b"".join(bytes.fromhex(ref.sha256) for ref in refs)).hexdigest()
    )


@pytest.mark.parametrize(
    "fault",
    [
        "absolute",
        "traversal",
        "symlink",
        "intermediate-symlink",
        "tampered",
        "short",
        "no-workspace",
    ],
)
async def test_workspace_csv_must_be_confined_and_match_its_reference(
    tmp_path: Path, fault: str
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    file = root / "invoices.csv"
    data = _CSV.encode()
    outside = tmp_path / "invoices.csv"
    outside.write_bytes(data)
    file.write_bytes(data)
    path = file.name
    if fault == "absolute":
        path = str(outside)
    elif fault == "traversal":
        path = f"../{outside.name}"
    elif fault == "symlink":
        file.unlink()
        file.symlink_to(outside)
    elif fault == "intermediate-symlink":
        (root / "external").symlink_to(tmp_path, target_is_directory=True)
        path = f"external/{outside.name}"
    elif fault == "tampered":
        file.write_bytes(data.replace(b"matched", b"altered"))
    elif fault == "short":
        file.write_bytes(data[:-1])

    events = Events()
    profile = specialist_profile("qc")
    driver = InProcessDriver({"qc": make_cutoff_qc_specialist()})
    workspace = None if fault == "no-workspace" else root
    async with driver.open(profile, workspace_path=workspace, access="read_only") as session:
        outcome = await session.run_turn(
            TurnRequest(
                run_id="blocked", prompt="Check the CSV", input_refs=(csv_ref(path, data),)
            ),
            events,
            deny_all,
        )
    assert isinstance(outcome, TurnFinished) and outcome.result.status == "failed"
    assert [event.code for event in events if isinstance(event, ErrorEvent)] == [
        "workspace_violation"
    ]
    assert isinstance(events[-1], FinishedEvent)


async def test_cancel_active_turn_sets_signal_and_releases_busy_session() -> None:
    began = asyncio.Event()
    stopped = asyncio.Event()
    contexts: list[SpecialistContext] = []

    async def handler(context: SpecialistContext, request: TurnRequest) -> TurnFinished:
        if request.prompt == "hold":
            contexts.append(context)
            began.set()
            try:
                await asyncio.Future()
            finally:
                stopped.set()
        return TurnFinished(result=TurnResult(run_id=request.run_id, status="completed"))

    driver = InProcessDriver({"example": handler})
    profile = specialist_profile("example")
    events = Events()
    async with (
        asyncio.timeout(3),
        driver.open(profile, workspace_path=None, access="read_only") as session,
    ):
        running = asyncio.create_task(
            session.run_turn(TurnRequest(run_id="active", prompt="hold"), events, deny_all)
        )
        await began.wait()
        await session.cancel("other")
        assert not contexts[0].cancelled.is_set()
        with pytest.raises(BusyError):
            await session.run_turn(
                TurnRequest(run_id="overlap", prompt="other"), Events(), deny_all
            )
        await session.cancel("active")
        assert contexts[0].cancelled.is_set()
        outcome = await running
        assert stopped.is_set()
        next_outcome = await session.run_turn(
            TurnRequest(run_id="next", prompt="continue"), Events(), deny_all
        )
    assert isinstance(outcome, TurnFinished)
    assert (outcome.result.status, outcome.result.reason) == ("cancelled", "cancelled_by_caller")
    assert events[-1] == FinishedEvent(
        run_id="active",
        seq=1,
        status="cancelled",
        reason="cancelled_by_caller",
        timestamp=events[-1].timestamp,
    )
    assert isinstance(next_outcome, TurnFinished) and next_outcome.result.status == "completed"


async def test_cancel_preserves_the_specialists_explicit_completion() -> None:
    started = asyncio.Event()
    events = Events()
    result = TurnResult(run_id="active", status="completed", text="The work was done")

    async def handler(context: SpecialistContext, request: TurnRequest) -> TurnFinished:
        started.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            assert context.cancelled.is_set()
            return TurnFinished(result=result)
        raise AssertionError("Expected cancellation")

    driver = InProcessDriver({"example": handler})
    async with (
        asyncio.timeout(3),
        driver.open(
            specialist_profile("example"), workspace_path=None, access="read_only"
        ) as session,
    ):
        running = asyncio.create_task(
            session.run_turn(TurnRequest(run_id="active", prompt="hold"), events, deny_all)
        )
        await started.wait()
        await session.cancel("active")
        assert await running == TurnFinished(result=result)
    assert [event.kind for event in events] == ["started", "finished"]
    assert isinstance(events[-1], FinishedEvent) and events[-1].status == "completed"


@pytest.mark.parametrize("interrupt", ["cancel", "close"])
async def test_stubborn_worker_times_out_and_keeps_session_busy(interrupt: str) -> None:
    started = asyncio.Event()
    cancellation_seen = asyncio.Event()
    release = asyncio.Event()
    worker: asyncio.Task[TurnOutcome] | None = None
    events = Events()

    async def handler(context: SpecialistContext, request: TurnRequest) -> TurnFinished:
        nonlocal worker
        if request.prompt == "hold":
            worker = asyncio.current_task()
            started.set()
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                cancellation_seen.set()
                await release.wait()
        return TurnFinished(result=TurnResult(run_id=request.run_id, status="completed"))

    profile = specialist_profile("example", close_timeout_s=0.03)
    driver = InProcessDriver({"example": handler})
    scope = driver.open(profile, workspace_path=None, access="read_only")
    session = await scope.__aenter__()
    running = asyncio.create_task(
        session.run_turn(TurnRequest(run_id="active", prompt="hold"), events, deny_all)
    )
    next_request = TurnRequest(run_id="next", prompt="run")
    closing = None
    try:
        async with asyncio.timeout(2):
            await started.wait()
            if interrupt == "close":
                closing = asyncio.create_task(scope.__aexit__(None, None, None))
                await asyncio.shield(closing)
            else:
                await session.cancel("active")
            outcome = await asyncio.shield(running)
        assert isinstance(outcome, TurnFinished)
        assert (outcome.result.status, outcome.result.reason) == ("indeterminate", "cancel_timeout")
        terminal = [event for event in events if isinstance(event, FinishedEvent)]
        assert [(event.status, event.reason) for event in terminal] == [
            ("indeterminate", "cancel_timeout")
        ]
        assert cancellation_seen.is_set() and worker is not None and not worker.done()
        with pytest.raises(BusyError):
            await session.run_turn(next_request, Events(), deny_all)
        release.set()
        await worker
        if closing is None:
            next_outcome = await session.run_turn(next_request, Events(), deny_all)
            assert (
                isinstance(next_outcome, TurnFinished) and next_outcome.result.status == "completed"
            )
        else:
            with pytest.raises(UnsupportedError, match="closed"):
                await session.run_turn(next_request, Events(), deny_all)
    finally:
        release.set()
        if closing is None:
            await scope.__aexit__(None, None, None)
        else:
            await closing
        await asyncio.gather(running, return_exceptions=True)


async def test_cancel_before_handler_start_still_emits_matching_terminal_event() -> None:
    started = asyncio.Event()
    release_start = asyncio.Event()
    events = Events()

    async def sink(event: RuntimeEvent) -> None:
        await events(event)
        if event.kind == "started":
            started.set()
            await release_start.wait()

    async def handler(context: SpecialistContext, request: TurnRequest) -> TurnFinished:
        pytest.fail("Handler ran after cancellation")

    driver = InProcessDriver({"example": handler})
    profile = specialist_profile("example")
    async with (
        asyncio.timeout(3),
        driver.open(profile, workspace_path=None, access="read_only") as session,
    ):
        running = asyncio.create_task(
            session.run_turn(TurnRequest(run_id="start", prompt="any"), sink, deny_all)
        )
        await started.wait()
        try:
            await session.cancel("start")
        finally:
            release_start.set()
        outcome = await running
    assert isinstance(outcome, TurnFinished) and outcome.result.status == "cancelled"
    assert [event.kind for event in events] == ["started", "finished"]


async def test_context_manager_drains_an_active_echo() -> None:
    driver = InProcessDriver({"echo": make_echo_specialist()})
    profile = specialist_profile("echo", slow=True)
    events = Events()
    started = asyncio.Event()

    async def sink(event: RuntimeEvent) -> None:
        await events(event)
        started.set()

    async with asyncio.timeout(3):
        async with driver.open(profile, workspace_path=None, access="read_only") as session:
            running = asyncio.create_task(
                session.run_turn(TurnRequest(run_id="hold", prompt="hello"), sink, deny_all)
            )
            await started.wait()
        outcome = await running
    assert isinstance(outcome, TurnFinished) and outcome.result.status == "cancelled"
    with pytest.raises(UnsupportedError, match="closed"):
        await session.run_turn(TurnRequest(run_id="closed", prompt="echo"), Events(), deny_all)


async def test_live_approval_is_correlated_and_echoes_after_allow() -> None:
    profile = specialist_profile("echo", approval_action="echo")
    driver = InProcessDriver({"echo": make_echo_specialist()})
    requests: list[ApprovalRequest] = []

    async def allow(request: ApprovalRequest) -> ApprovalDecision:
        requests.append(request)
        return ApprovalDecision(outcome="allow_once", reason="Allowed by test policy")

    events = Events()
    async with driver.open(profile, workspace_path=None, access="read_only") as session:
        assert session.capabilities.approvals == "live"
        outcome = await session.run_turn(
            TurnRequest(run_id="approval", prompt="hello"), events, allow
        )
    assert isinstance(outcome, TurnFinished) and outcome.result.text == "hello"
    (request,) = requests
    assert request.action == "echo" and request.run_id == "approval"
    assert "allow_once" in request.offered_options
    asked, decided = [event for event in events if isinstance(event, ApprovalEvent)]
    assert asked.request_id == decided.request_id == request.request_id
    assert (asked.phase, decided.phase, decided.outcome) == ("requested", "decided", "allow_once")
    assert [event.seq for event in events] == list(range(5))


@pytest.mark.parametrize("interrupt", ["cancel", "deadline"])
async def test_interrupted_turn_discards_late_approval(interrupt: str) -> None:
    waiting_for_policy = asyncio.Event()
    events = Events()

    async def late_policy(request: ApprovalRequest) -> ApprovalDecision:
        waiting_for_policy.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            return ApprovalDecision(outcome="allow_once")
        raise AssertionError("The policy must wait for cancellation")

    profile = specialist_profile("echo", approval_action="echo")
    driver = InProcessDriver({"echo": make_echo_specialist()})
    async with (
        asyncio.timeout(3),
        driver.open(profile, workspace_path=None, access="read_only") as session,
    ):
        request = TurnRequest(
            run_id="approval", prompt="hello", timeout_s=0.2 if interrupt == "deadline" else None
        )
        running = asyncio.create_task(session.run_turn(request, events, late_policy))
        await waiting_for_policy.wait()
        if interrupt == "cancel":
            await session.cancel(request.run_id)
        outcome = await running
    asked, decided = [event for event in events if isinstance(event, ApprovalEvent)]
    assert asked.request_id == decided.request_id and decided.outcome == "cancelled"
    assert all(event.kind != "text" for event in events)
    assert isinstance(outcome, TurnFinished)
    assert outcome.result.status == ("cancelled" if interrupt == "cancel" else "failed")


async def test_policy_exception_denies_without_publishing_exception_text() -> None:
    async def broken_policy(request: ApprovalRequest) -> ApprovalDecision:
        raise RuntimeError("approval token=fixture-secret")

    profile = specialist_profile("echo", approval_action="echo")
    driver = InProcessDriver({"echo": make_echo_specialist()})
    events = Events()
    async with driver.open(profile, workspace_path=None, access="read_only") as session:
        outcome = await session.run_turn(
            TurnRequest(run_id="policy-error", prompt="hello"), events, broken_policy
        )
    assert isinstance(outcome, TurnFinished) and outcome.result.reason == "approval_denied"
    assert [event.outcome for event in events if isinstance(event, ApprovalEvent)] == [None, "deny"]
    assert "fixture-secret" not in str(events)


async def test_direct_handler_has_session_context_and_can_emit_concurrently(tmp_path: Path) -> None:
    events = Events()

    async def delayed_sink(event: RuntimeEvent) -> None:
        if event.kind == "usage":
            await asyncio.sleep(0)
        await events(event)

    async def handler(context: SpecialistContext, request: TurnRequest) -> TurnFinished:
        assert context.profile.name == "direct"
        assert context.workspace_path == tmp_path
        assert context.access == "read_only" and context.resume_ref == "saved"
        await asyncio.gather(
            context.emit(lambda **_: UsageEvent(run_id="old", seq=2, input_tokens=2)),
            context.emit(TextEvent(run_id="replace", seq=99, text="finished")),
        )
        return TurnFinished(
            result=TurnResult(run_id=request.run_id, status="completed", native_session_ref="next")
        )

    profile = AgentProfile(
        name="direct",
        kind="inprocess",
        options={"handler": handler},
        capabilities=Capabilities(resume=True, enforced_sandbox={"read_only"}),
    )
    async with InProcessDriver().open(
        profile, workspace_path=tmp_path, access="read_only", resume_ref="saved"
    ) as session:
        outcome = await session.run_turn(
            TurnRequest(run_id="resume", prompt="example"), delayed_sink, deny_all
        )
        assert session.native_ref == "next"
    assert isinstance(outcome, TurnFinished) and outcome.result.status == "completed"
    assert [event.kind for event in events] == ["started", "usage", "text", "finished"]
    assert [event.seq for event in events] == [0, 1, 2, 3]
    assert {event.run_id for event in events} == {"resume"}


@pytest.mark.parametrize("failure", ["exception", "duplicate-terminal"])
async def test_failed_handler_publishes_error_and_finished_without_raw_exception(
    failure: str,
) -> None:
    async def broken_handler(context: SpecialistContext, request: TurnRequest) -> TurnFinished:
        if failure == "duplicate-terminal":
            await context.emit(FinishedEvent, status="completed")
        raise RuntimeError("token=fixture-secret")

    profile = specialist_profile("broken")
    driver = InProcessDriver({"broken": broken_handler})
    events = Events()
    async with driver.open(profile, workspace_path=None, access="read_only") as session:
        outcome = await session.run_turn(
            TurnRequest(run_id="failed", prompt="example"), events, deny_all
        )
    assert isinstance(outcome, TurnFinished) and outcome.result.status == "failed"
    assert [event.kind for event in events] == ["started", "error", "finished"]
    assert isinstance(events[1], ErrorEvent)
    assert events[1].code == ("specialist_error" if failure == "exception" else "invalid_output")
    assert "fixture-secret" not in str(events) and "fixture-secret" not in outcome.result.reason


async def test_timeout_cancels_slow_echo() -> None:
    profile = specialist_profile("echo")
    driver = InProcessDriver({"echo": make_echo_specialist()})
    events = Events()
    async with (
        asyncio.timeout(3),
        driver.open(profile, workspace_path=None, access="read_only") as session,
    ):
        outcome = await session.run_turn(
            TurnRequest(run_id="timeout", prompt="slow", timeout_s=0.01), events, deny_all
        )
    assert isinstance(outcome, TurnFinished) and outcome.result.status == "failed"
    assert [event.code for event in events if isinstance(event, ErrorEvent)] == ["turn_timeout"]


async def test_open_rejects_an_unknown_handler_and_unsupported_scope() -> None:
    profile = specialist_profile("echo")
    driver = InProcessDriver({"echo": make_echo_specialist()})
    unknown = profile.model_copy(update={"options": {"handler": "unknown"}})
    with pytest.raises(UnsupportedError, match="registered"):
        async with driver.open(unknown, workspace_path=None, access="read_only"):
            pytest.fail("Opened an unknown handler")
    with pytest.raises(UnsupportedError, match="workspace scope"):
        async with driver.open(profile, workspace_path=None, access="read_write"):
            pytest.fail("Opened a writable in-process workspace")
    profile_without_scope = AgentProfile(name="echo", kind="inprocess", options={"handler": "echo"})
    assert profile_without_scope.capabilities.enforced_sandbox == frozenset()
    with pytest.raises(UnsupportedError, match="workspace scope"):
        async with driver.open(profile_without_scope, workspace_path=None, access="read_only"):
            pytest.fail("Opened a workspace scope excluded by its profile")
    with pytest.raises(ResumeUnavailableError):
        async with driver.open(
            profile, workspace_path=None, access="read_only", resume_ref="saved"
        ):
            pytest.fail("Opened an unsupported native resume reference")


@pytest.mark.parametrize("name", ["echo", "invoice-qc"])
async def test_default_registry_warns_when_selecting_a_demo_by_name(name: str) -> None:
    profile = AgentProfile(
        name=name,
        kind="inprocess",
        capabilities=Capabilities(enforced_sandbox={"read_only"}),
    )
    with pytest.warns(UserWarning, match="Set options.handler explicitly"):
        async with default_drivers()["inprocess"].open(
            profile, workspace_path=None, access="read_only"
        ):
            pass


async def test_read_only_turn_rejects_file_outputs_before_dispatch() -> None:
    profile = specialist_profile("echo")
    driver = InProcessDriver({"echo": make_echo_specialist()})
    events = Events()
    async with driver.open(profile, workspace_path=None, access="read_only") as session:
        with pytest.raises(UnsupportedError, match="read-only"):
            await session.run_turn(
                TurnRequest(
                    run_id="no-write",
                    prompt="create file",
                    outputs=(OutputSpec(name="o", path="out"),),
                ),
                events,
                deny_all,
            )
    assert not events
