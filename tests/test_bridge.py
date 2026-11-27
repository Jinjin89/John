from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from a2a.helpers.proto_helpers import new_data_part
from a2a.server.agent_execution import SimpleRequestContextBuilder
from a2a.server.context import ServerCallContext
from a2a.server.events import Event, EventQueue
from a2a.server.events.event_queue_v2 import EventQueueSource
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.types import (
    CancelTaskRequest,
    GetTaskRequest,
    Message,
    Part,
    Role,
    SendMessageConfiguration,
    SendMessageRequest,
    Task,
    TaskState,
    TaskStatusUpdateEvent,
)
from a2a.utils.errors import ExtensionSupportRequiredError, InternalError, InvalidParamsError
from google.protobuf.json_format import MessageToDict

from john.bridge import JohnAgentExecutor, build_agent_card
from john.ledger import SqliteLedger
from john.runtime import SessionRuntime
from john.types import (
    LIFECYCLE_EXT_URI,
    WORKSPACE_EXT_URI,
    AgentProfile,
    ApprovalDecision,
    ApprovalPolicy,
    ApprovalRequest,
    Capabilities,
    EventSink,
    FinishedEvent,
    OutputSpec,
    SkillSpec,
    StartedEvent,
    TurnFinished,
    TurnNeedsInput,
    TurnOutcome,
    TurnRequest,
    TurnResult,
    WorkspaceAccess,
    WorkspaceGrant,
)
from john.workspace import WorkspaceRegistry, build_workspace_artifact, verify_workspace_artifact

SUBMITTED = TaskState.TASK_STATE_SUBMITTED
WORKING = TaskState.TASK_STATE_WORKING
INPUT_REQUIRED = TaskState.TASK_STATE_INPUT_REQUIRED
COMPLETED = TaskState.TASK_STATE_COMPLETED
CANCELED = TaskState.TASK_STATE_CANCELED
FAILED = TaskState.TASK_STATE_FAILED
REJECTED = TaskState.TASK_STATE_REJECTED

type Program = Callable[[TurnRequest, EventSink, ApprovalPolicy], Awaitable[TurnOutcome]]


class FixtureDriver:
    native_ref = "native-ref"
    ignored_fields: dict[str, str] = {}

    def __init__(self, program: Program, capabilities: Capabilities | None = None):
        self.program = program
        self.capabilities = capabilities or Capabilities()
        self.opened: list[tuple[Path | None, WorkspaceAccess, str | None]] = []
        self.requests: list[TurnRequest] = []
        self.cancel_calls: list[str] = []
        self.cancellation = asyncio.Event()

    @asynccontextmanager
    async def open(
        self,
        profile: AgentProfile,
        *,
        workspace_path: Path | None,
        access: WorkspaceAccess,
        resume_ref: str | None = None,
    ) -> AsyncIterator[FixtureDriver]:
        self.opened.append((workspace_path, access, resume_ref))
        yield self

    async def run_turn(
        self, request: TurnRequest, on_event: EventSink, on_approval: ApprovalPolicy
    ) -> TurnOutcome:
        self.requests.append(request)
        return await self.program(request, on_event, on_approval)

    async def cancel(self, run_id: str) -> None:
        self.cancel_calls.append(run_id)
        self.cancellation.set()


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "ledger.db"


@pytest.fixture
async def ledger(db_path: Path) -> AsyncIterator[SqliteLedger]:
    opened = await SqliteLedger.open(db_path)
    try:
        yield opened
    finally:
        await opened.close()


def owner(context: ServerCallContext) -> str:
    return "local"


def profile(**options) -> AgentProfile:
    return AgentProfile(name="review", kind="test", **options)


def context(*, workspace: bool = False, **state) -> ServerCallContext:
    return ServerCallContext(
        requested_extensions={WORKSPACE_EXT_URI} if workspace else set(), state=state
    )


def request(
    *,
    text: str = "Review the sample",
    task: Task | None = None,
    task_context_id: str | None = None,
    workspace: dict | None = None,
    extra_parts: tuple[Part, ...] = (),
    immediate: bool = False,
) -> SendMessageRequest:
    message = Message(
        message_id="follow-up" if task is not None else "new-message",
        role=Role.ROLE_USER,
        parts=[Part(text=text, media_type="text/plain"), *extra_parts],
        metadata={WORKSPACE_EXT_URI: workspace} if workspace is not None else None,
    )
    if task is not None:
        message.task_id = task.id
        message.context_id = task_context_id or task.context_id
    return SendMessageRequest(
        message=message, configuration=SendMessageConfiguration(return_immediately=immediate)
    )


@asynccontextmanager
async def handler_for(
    ledger: SqliteLedger,
    spec: AgentProfile,
    driver: FixtureDriver,
    registry: WorkspaceRegistry | None = None,
    *,
    cancel_timeout_s: float = 0.05,
) -> AsyncIterator[DefaultRequestHandler]:
    executor = JohnAgentExecutor(
        spec,
        driver,
        ledger,
        registry or WorkspaceRegistry(),
        SessionRuntime(ledger),
        owner_resolver=owner,
        cancel_timeout_s=cancel_timeout_s,
    )
    with pytest.warns(UserWarning, match="cross-replica streaming is disabled"):
        handler = DefaultRequestHandler(
            agent_executor=executor,
            task_store=ledger.task_store_for(spec.name, owner_resolver=owner),
            agent_card=build_agent_card(spec),
        )
    try:
        yield handler
    finally:
        await handler.aclose()


async def submit(
    handler: DefaultRequestHandler, params: SendMessageRequest, call: ServerCallContext
) -> Task:
    task = await asyncio.wait_for(handler.on_message_send(params, call), 5)
    assert isinstance(task, Task)
    return task


async def completed(request: TurnRequest, emit: EventSink, approve: ApprovalPolicy) -> TurnOutcome:
    return TurnFinished(result=TurnResult(run_id=request.run_id, status="completed", text="ok"))


def registry_for(path: Path, *, principal: str = "local") -> WorkspaceRegistry:
    registry = WorkspaceRegistry()
    registry.register(
        WorkspaceGrant(
            root_id="shared",
            host_path=path,
            max_access="read_write",
            allowed_principals=frozenset({principal}),
            allowed_agents=frozenset({"review"}),
        )
    )
    return registry


def test_agent_card_reflects_profile_skills_and_workspace_capabilities() -> None:
    capability = Capabilities(
        streaming=False, enforced_sandbox=frozenset({"read_write", "read_only"})
    )
    spec = profile(
        description="Review sample data.",
        require_workspace=True,
        capabilities=capability,
        skills=(SkillSpec(id="sample", name="Sample review", tags=("review",)),),
    )
    card = build_agent_card(spec, url="https://example.invalid/a2a")
    (interface,) = card.supported_interfaces
    assert (interface.url, interface.protocol_binding, interface.protocol_version) == (
        "https://example.invalid/a2a",
        "JSONRPC",
        "1.0",
    )
    assert not card.capabilities.streaming and not card.capabilities.push_notifications
    (extension,) = card.capabilities.extensions
    assert (extension.uri, extension.required) == (WORKSPACE_EXT_URI, True)
    assert MessageToDict(extension.params) == {
        "enforcedSandbox": ["read_only", "read_write"],
        "defaultAccess": "read_only",
    }
    (skill,) = card.skills
    assert (skill.id, skill.name, list(skill.tags)) == ("sample", "Sample review", ["review"])
    (default_skill,) = build_agent_card(profile(description="Review samples.")).skills
    assert (default_skill.id, default_skill.description) == ("review", "Review samples.")


async def test_workspace_handoff_artifact_and_terminal_commit_through_v2(
    ledger: SqliteLedger, db_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def disallow_temp_copy(*args, **kwargs):
        pytest.fail("Input reference validation must hash the file in place")

    monkeypatch.setattr("john.workspace.TemporaryFile", disallow_temp_copy)
    (tmp_path / "input.txt").write_text("sample", encoding="utf-8")
    registry = registry_for(tmp_path)
    incoming = build_workspace_artifact(
        "shared", OutputSpec(name="input", path="input.txt"), registry, agent_name="review"
    )
    workspace = {
        "root": "shared",
        "access": "read_write",
        "outputs": [{"name": "report", "path": "report.txt", "mediaType": "text/plain"}],
    }
    caps = Capabilities(enforced_sandbox=frozenset({"read_write"}))
    spec = profile(capabilities=caps, require_workspace=True)
    observed = []
    decision_seen = asyncio.Event()

    async def allow(req: ApprovalRequest) -> ApprovalDecision:
        return ApprovalDecision(outcome="allow_once", reason="explicit approval")

    async def observe(event) -> None:
        observed.append(event)
        if event.kind == "approval" and event.phase == "decided":
            decision_seen.set()

    async def program(turn: TurnRequest, emit: EventSink, approve: ApprovalPolicy) -> TurnOutcome:
        (ref,) = turn.input_refs
        assert (ref.root, ref.path) == ("shared", "input.txt")
        assert turn.outputs[0].path == "report.txt"
        await emit(StartedEvent(run_id=turn.run_id, seq=12, native_session_ref="native"))
        approval = ApprovalRequest(request_id="write-report", run_id=turn.run_id, action="write")
        assert (await approve(approval)).outcome == "allow_once"
        (tmp_path / "report.txt").write_text("verified output", encoding="utf-8")
        await emit(FinishedEvent(run_id=turn.run_id, seq=50, status="completed"))
        return TurnFinished(
            result=TurnResult(
                run_id=turn.run_id,
                status="completed",
                native_session_ref="native",
                usage={"tokens": 2},
            )
        )

    driver = FixtureDriver(program, caps)
    call = context(workspace=True, john_policy=allow, john_on_event=observe)
    observer = await SqliteLedger.open(db_path, exclusive=False)
    try:
        async with handler_for(ledger, spec, driver, registry) as handler:
            events = []
            async with asyncio.timeout(5):
                async for event in handler.on_message_send_stream(
                    request(workspace=workspace, extra_parts=(incoming.parts[0],)), call
                ):
                    events.append(event)
                    if isinstance(event, TaskStatusUpdateEvent) and event.status.state == WORKING:
                        stored = await observer.task_store_for("review", owner_resolver=owner).get(
                            event.task_id, call
                        )
                        assert stored is not None
                        if stored.task.status.state == WORKING:
                            assert not stored.task.artifacts
                    if isinstance(event, TaskStatusUpdateEvent) and event.status.state == COMPLETED:
                        stored = await observer.task_store_for("review", owner_resolver=owner).get(
                            event.task_id, call
                        )
                        assert stored is not None and stored.task.status.state == COMPLETED
                        (run,) = await observer.list_runs(task_id=event.task_id)
                        assert run["status"] == "completed" and run["usage"] == {"tokens": 2}
                        journal = await observer.get_run_events(run["run_id"])
                        assert [(entry.kind, entry.seq) for entry in journal] == [
                            ("started", 0),
                            ("approval", 1),
                            ("approval", 2),
                            ("finished", 3),
                        ]
                        (artifact,) = stored.task.artifacts
                        ref, content = verify_workspace_artifact(
                            artifact, registry, principal="local", agent_name="review"
                        )
                        assert ref.path == "report.txt" and content == b"verified output"
                await decision_seen.wait()
            assert driver.opened == [(tmp_path, "read_write", None)]
            assert isinstance(events[0], Task) and events[0].status.state == SUBMITTED
            assert (
                isinstance(events[1], TaskStatusUpdateEvent) and events[1].status.state == WORKING
            )
            assert events[-1].status.state == COMPLETED
            assert observed[-1].outcome == "allow_once"
    finally:
        await observer.close()


async def test_missing_required_extension_is_an_a2a_error(ledger: SqliteLedger) -> None:
    driver = FixtureDriver(completed)
    async with handler_for(ledger, profile(require_workspace=True), driver) as handler:
        with pytest.raises(ExtensionSupportRequiredError, match="required workspace extension"):
            await submit(handler, request(), context())
    assert driver.opened == [] and await ledger.list_runs() == []


@pytest.mark.parametrize(
    ("problem", "expected_opens", "message"),
    [
        ("principal", 0, "unauthorized"),
        ("profile", 0, "Driver lacks sandbox enforcement"),
        ("session", 1, "Session cannot enforce"),
        ("untyped-part", 0, "Input part is not supported"),
        ("digest", 0, "SHA-256 mismatch"),
    ],
)
async def test_invalid_workspace_or_input_is_rejected_before_dispatch(
    ledger: SqliteLedger, tmp_path: Path, problem: str, expected_opens: int, message: str
) -> None:
    registry = registry_for(tmp_path, principal="other" if problem == "principal" else "local")
    profile_caps = Capabilities(
        enforced_sandbox=frozenset({"read_only" if problem == "profile" else "read_write"})
    )
    session_caps = Capabilities(
        enforced_sandbox=frozenset({"read_only" if problem == "session" else "read_write"})
    )
    extra_parts: tuple[Part, ...] = ()
    if problem == "untyped-part":
        extra_parts = (new_data_part({"ignored": "must fail"}, media_type="application/json"),)
    if problem == "digest":
        target = tmp_path / "sample.txt"
        target.write_bytes(b"old")
        artifact = build_workspace_artifact(
            "shared", OutputSpec(name="sample", path=target.name), registry, agent_name="review"
        )
        target.write_bytes(b"new")
        extra_parts = (artifact.parts[0],)
    driver = FixtureDriver(completed, session_caps)
    async with handler_for(ledger, profile(capabilities=profile_caps), driver, registry) as handler:
        task = await submit(
            handler,
            request(workspace={"root": "shared", "access": "read_write"}, extra_parts=extra_parts),
            context(workspace=True),
        )
    assert task.status.state == REJECTED and message in task.status.message.parts[0].text
    assert len(driver.opened) == expected_opens and driver.requests == []
    assert await ledger.list_runs() == []


async def test_missing_declared_output_fails_without_publishing_partial_artifacts(
    ledger: SqliteLedger, tmp_path: Path
) -> None:
    async def write_one_output(turn: TurnRequest, emit: EventSink, approve: ApprovalPolicy):
        (tmp_path / "created.txt").write_text("created", encoding="utf-8")
        return await completed(turn, emit, approve)

    caps = Capabilities(enforced_sandbox=frozenset({"read_write"}))
    registry = registry_for(tmp_path)
    driver = FixtureDriver(write_one_output, caps)
    workspace = {
        "root": "shared",
        "access": "read_write",
        "outputs": [
            {"name": "valid", "path": "created.txt"},
            {"name": "missing", "path": "missing.txt"},
        ],
    }
    async with handler_for(ledger, profile(capabilities=caps), driver, registry) as handler:
        task = await submit(handler, request(workspace=workspace), context(workspace=True))
    assert task.status.state == FAILED and not task.artifacts
    assert task.status.message.parts[0].text.startswith("invalid_output:")
    (run,) = await ledger.list_runs(task_id=task.id)
    (finished,) = await ledger.get_run_events(run["run_id"])
    assert run["status"] == "failed" and finished.status == "failed"


async def test_existing_declared_output_is_rejected_before_driver_open(
    ledger: SqliteLedger, tmp_path: Path
) -> None:
    (tmp_path / "stale.txt").write_text("earlier result", encoding="utf-8")
    caps = Capabilities(enforced_sandbox=frozenset({"read_write"}))
    driver = FixtureDriver(completed, caps)
    workspace = {
        "root": "shared",
        "access": "read_write",
        "outputs": [{"name": "report", "path": "stale.txt"}],
    }
    async with handler_for(
        ledger, profile(capabilities=caps), driver, registry_for(tmp_path)
    ) as handler:
        task = await submit(handler, request(workspace=workspace), context(workspace=True))
    assert task.status.state == REJECTED
    assert task.status.message.parts[0].text == "Declared output path already exists"
    assert not driver.opened and await ledger.list_runs() == []


@pytest.mark.parametrize("advertised_by", ["profile", "skill", "neither"])
async def test_structured_input_is_added_to_the_prompt_when_advertised(
    ledger: SqliteLedger, advertised_by: str
) -> None:
    fields = {}
    if advertised_by == "profile":
        fields["default_input_modes"] = ("application/json",)
    elif advertised_by == "skill":
        fields["skills"] = (
            SkillSpec(id="review", name="Review", input_modes=("application/json",)),
        )
    part = new_data_part({"z": [True, "x"], "a": {"b": False}}, media_type="application/json")
    driver = FixtureDriver(completed)
    async with handler_for(ledger, profile(**fields), driver) as handler:
        task = await submit(handler, request(text="Review", extra_parts=(part,)), context())
    if advertised_by == "neither":
        assert task.status.state == REJECTED and not driver.opened
    else:
        assert task.status.state == COMPLETED
        assert driver.requests[0].prompt == 'Review\n{"a":{"b":false},"z":[true,"x"]}'


async def test_pause_and_resume_after_reopening_ledger(
    ledger: SqliteLedger, db_path: Path, tmp_path: Path
) -> None:
    caps = Capabilities(task_questions=True, resume=True, enforced_sandbox=frozenset({"read_only"}))
    spec = profile(capabilities=caps, require_workspace=True)
    registry = registry_for(tmp_path)
    schema = {"type": "integer", "minimum": 0}

    async def pause(turn: TurnRequest, emit: EventSink, approve: ApprovalPolicy) -> TurnOutcome:
        assert turn.schema_def == schema and turn.continuation is None
        await emit(StartedEvent(run_id=turn.run_id, seq=0))
        return TurnNeedsInput(
            run_id=turn.run_id,
            question="Which cutoff?",
            continuation={"batch": 17},
            native_session_ref="paused-session",
        )

    first_driver = FixtureDriver(pause, caps)
    async with handler_for(ledger, spec, first_driver, registry) as handler:
        paused = await submit(
            handler,
            request(workspace={"root": "shared"}),
            context(workspace=True, john_schema=schema),
        )
        assert paused.status.state == INPUT_REQUIRED
        assert paused.status.message.parts[0].text == "Which cutoff?"
        (first_run,) = await ledger.list_runs(task_id=paused.id)
        assert first_run["status"] == "incomplete" and first_run["reason"] == "input_required"
    await ledger.close()

    async def resume(turn: TurnRequest, emit: EventSink, approve: ApprovalPolicy) -> TurnOutcome:
        assert turn.continuation == {"batch": 17} and turn.schema_def == schema
        assert turn.prompt == "5"
        return TurnFinished(result=TurnResult(run_id=turn.run_id, status="completed", text="5"))

    reopened = await SqliteLedger.open(db_path)
    try:
        next_driver = FixtureDriver(resume, caps)
        async with handler_for(reopened, spec, next_driver, registry) as handler:
            assert await reopened.recover_crashed_runs() == []
            with pytest.raises(InvalidParamsError, match="does not match"):
                await submit(
                    handler, request(task=paused, task_context_id="wrong", text="5"), context()
                )
            for workspace, extensions in ((None, False), ({"root": "different"}, True)):
                rejected_reply = await submit(
                    handler,
                    request(task=paused, text="5", workspace=workspace),
                    context(workspace=extensions),
                )
                assert rejected_reply.status.state == INPUT_REQUIRED
                assert rejected_reply.status.message.parts[0].text == "Which cutoff?"
                checkpoint = await reopened.peek_continuation(
                    "local", "review", paused.id, paused.context_id
                )
                assert checkpoint is not None and checkpoint["run_id"] == first_run["run_id"]
                assert not next_driver.opened
            result = await submit(
                handler,
                request(task=paused, text="5"),
                context(workspace=True, john_schema={"type": "string"}),
            )
            assert result.status.state == COMPLETED
            assert (result.id, result.context_id) == (paused.id, paused.context_id)
            assert next_driver.opened == [(tmp_path, "read_only", "paused-session")]
            part = result.artifacts[0].parts[0]
            assert part.media_type == "application/json" and MessageToDict(part.data) == 5
            runs = await reopened.list_runs(task_id=result.id)
            assert [run["status"] for run in runs] == ["incomplete", "completed"]
            assert runs[1]["parent_run_id"] == runs[0]["run_id"]
            assert (
                await reopened.consume_continuation("local", "review", result.id, result.context_id)
                is None
            )
    finally:
        await reopened.close()


@pytest.mark.parametrize(
    ("failed_state", "closed"), [(COMPLETED, False), (INPUT_REQUIRED, False), (COMPLETED, True)]
)
async def test_publish_queue_failure_records_indeterminate_outcome(
    ledger: SqliteLedger,
    failed_state: TaskState.ValueType,
    closed: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    enqueue = EventQueueSource.enqueue_event
    is_closed = EventQueueSource.is_closed
    closed_queue = None

    async def fail_final_enqueue(queue: EventQueueSource, event: Event) -> None:
        nonlocal closed_queue
        if isinstance(event, TaskStatusUpdateEvent) and event.status.state == failed_state:
            if closed:
                closed_queue = queue
                return
            raise RuntimeError("injected publication failure")
        await enqueue(queue, event)

    async def program(turn: TurnRequest, emit: EventSink, approve: ApprovalPolicy) -> TurnOutcome:
        if failed_state == INPUT_REQUIRED:
            return TurnNeedsInput(run_id=turn.run_id, question="Which cutoff?", continuation={})
        return await completed(turn, emit, approve)

    monkeypatch.setattr(EventQueueSource, "enqueue_event", fail_final_enqueue)
    monkeypatch.setattr(
        EventQueueSource, "is_closed", lambda queue: queue is closed_queue or is_closed(queue)
    )
    caps = Capabilities(task_questions=True)
    driver = FixtureDriver(program, caps)
    async with handler_for(ledger, profile(capabilities=caps), driver) as handler:
        if closed:
            try:
                returned = await submit(handler, request(), context())
            except InternalError as exc:
                assert "publication" in str(exc)
            else:
                assert returned.status.state == FAILED and not returned.artifacts
                assert MessageToDict(returned.metadata) == {
                    LIFECYCLE_EXT_URI: {"indeterminate": True}
                }
        else:
            with pytest.raises(RuntimeError, match="publication"):
                await submit(handler, request(), context())
        (run,) = await ledger.list_runs()
        saved = await handler.on_get_task(GetTaskRequest(id=run["task_id"]), context())
        assert saved is not None and saved.status.state == FAILED and not saved.artifacts
        assert MessageToDict(saved.metadata) == {LIFECYCLE_EXT_URI: {"indeterminate": True}}
        assert run["status"] == "indeterminate" and run["reason"] == "publish_failed"
        assert (await ledger.get_run_events(run["run_id"]))[-1].status == "indeterminate"
        assert await ledger.peek_continuation("local", "review", saved.id, saved.context_id) is None


async def test_duplicate_execution_raises_typed_a2a_error(ledger: SqliteLedger) -> None:
    started, release = asyncio.Event(), asyncio.Event()

    async def hold(turn: TurnRequest, emit: EventSink, approve: ApprovalPolicy) -> TurnOutcome:
        started.set()
        await release.wait()
        return await completed(turn, emit, approve)

    executor = JohnAgentExecutor(
        profile(), FixtureDriver(hold), ledger, WorkspaceRegistry(), SessionRuntime(ledger)
    )
    request_context = await SimpleRequestContextBuilder().build(
        params=request(), task_id="task", context_id="context", context=context()
    )

    class DiscardEvents(EventQueue):
        async def enqueue_event(self, event: Event) -> None:
            pass

    first = asyncio.create_task(executor.execute(request_context, DiscardEvents()))
    try:
        await asyncio.wait_for(started.wait(), 5)
        with pytest.raises(InvalidParamsError, match="Task already has an active execution"):
            await executor.execute(request_context, DiscardEvents())
    finally:
        release.set()
        await first


@pytest.mark.parametrize(
    ("native_result", "task_state", "run_state"),
    [
        ("cancelled", CANCELED, "cancelled"),
        ("completed", COMPLETED, "completed"),
        ("unconfirmed", FAILED, "indeterminate"),
    ],
)
async def test_v2_cancel_waits_for_the_authoritative_outcome(
    ledger: SqliteLedger, native_result: str, task_state: TaskState.ValueType, run_state: str
) -> None:
    started = asyncio.Event()

    async def program(turn: TurnRequest, emit: EventSink, approve: ApprovalPolicy) -> TurnOutcome:
        await emit(StartedEvent(run_id=turn.run_id, seq=0))
        started.set()
        await driver.cancellation.wait()
        if native_result == "unconfirmed":
            await asyncio.Future()
        return TurnFinished(
            result=TurnResult(run_id=turn.run_id, status=native_result, text="result")
        )

    driver = FixtureDriver(program)
    call = context()
    async with handler_for(ledger, profile(), driver) as handler:
        initial = await submit(handler, request(immediate=True), call)
        async with asyncio.timeout(5):
            await started.wait()
            final = await handler.on_cancel_task(CancelTaskRequest(id=initial.id), call)
            persisted = await handler.on_get_task(GetTaskRequest(id=initial.id), call)
        assert final is not None and final.status.state == task_state and persisted == final
        (run,) = await ledger.list_runs(task_id=initial.id)
        assert run["status"] == run_state and driver.cancel_calls == [run["run_id"]]
        events = await ledger.get_run_events(run["run_id"])
        assert [event.kind for event in events] == ["started", "finished"]
        assert events[-1].status == run_state
        if native_result == "unconfirmed":
            assert MessageToDict(final.metadata) == {LIFECYCLE_EXT_URI: {"indeterminate": True}}
            assert run["reason"] == final.status.message.parts[0].text == "cancel_timeout"
        if native_result == "completed":
            assert final.artifacts[0].parts[0].text == "result"


async def test_cancelling_a_parked_task_clears_its_continuation(ledger: SqliteLedger) -> None:
    caps = Capabilities(task_questions=True)

    async def pause(turn: TurnRequest, emit: EventSink, approve: ApprovalPolicy) -> TurnOutcome:
        return TurnNeedsInput(run_id=turn.run_id, question="Which cutoff?")

    driver = FixtureDriver(pause, caps)
    call = context()
    async with handler_for(ledger, profile(capabilities=caps), driver) as handler:
        paused = await submit(handler, request(), call)
        assert paused.status.state == INPUT_REQUIRED
        cancelled = await asyncio.wait_for(
            handler.on_cancel_task(CancelTaskRequest(id=paused.id), call), 5
        )
        assert cancelled is not None and cancelled.status.state == CANCELED
    assert (
        await ledger.consume_continuation("local", "review", paused.id, paused.context_id) is None
    )
    assert driver.cancel_calls == []
