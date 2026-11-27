import asyncio
import sqlite3
from collections.abc import AsyncIterator
from contextlib import closing
from pathlib import Path

import pytest
from a2a.auth.user import User
from a2a.helpers.proto_helpers import new_data_part
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.cluster.task_store import ConcurrentTaskModificationError, VersionedTaskStore
from a2a.server.cluster.version import TaskVersion
from a2a.server.context import ServerCallContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler, DefaultRequestHandlerV2
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentSkill,
    Artifact,
    GetTaskRequest,
    ListTasksRequest,
    Message,
    Part,
    Role,
    SendMessageRequest,
    Task,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
)
from a2a.utils.errors import InvalidParamsError, TaskNotFoundError
from a2a.utils.task import encode_page_token
from google.protobuf.json_format import MessageToDict

from john.ledger import SqliteLedger
from john.types import (
    LIFECYCLE_EXT_URI,
    BusyError,
    DuplicateMessageConflictError,
    FinishedEvent,
    StartedEvent,
    TextEvent,
    WorkspaceRequest,
)

WORKING = TaskState.TASK_STATE_WORKING
INPUT_REQUIRED = TaskState.TASK_STATE_INPUT_REQUIRED
COMPLETED = TaskState.TASK_STATE_COMPLETED
FAILED = TaskState.TASK_STATE_FAILED
CANCELED = TaskState.TASK_STATE_CANCELED
SUBMITTED = TaskState.TASK_STATE_SUBMITTED


class Principal(User):
    def __init__(self, name: str):
        self.name = name

    @property
    def is_authenticated(self) -> bool:
        return True

    @property
    def user_name(self) -> str:
        return self.name


def caller(name: str = "alice") -> ServerCallContext:
    return ServerCallContext(user=Principal(name))


def make_task(
    task_id: str = "task",
    *,
    context_id: str = "context",
    state: TaskState.ValueType = WORKING,
    timestamp_ns: int | None = None,
) -> Task:
    task = Task(id=task_id, context_id=context_id, status=TaskStatus(state=state))
    if timestamp_ns is not None:
        task.status.timestamp.FromNanoseconds(timestamp_ns)
    return task


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


async def test_exclusive_execution_lock_and_full_sync(ledger: SqliteLedger, db_path: Path) -> None:
    assert await ledger._call(lambda: ledger._db.execute("PRAGMA synchronous").fetchone()[0]) == 2
    with pytest.raises(BusyError, match="Ledger is locked by another active coordinator"):
        await SqliteLedger.open(db_path)
    inspection = await SqliteLedger.open(db_path, exclusive=False)
    try:
        assert await inspection.list_runs() == []
        with pytest.raises(BusyError, match="requires exclusive ledger ownership"):
            await inspection.recover_crashed_runs()
        with pytest.raises(BusyError, match="Dispatch requires exclusive ledger ownership"):
            await inspection.start_run(
                "run", owner="alice", agent_name="review", task_id="task", context_id="context"
            )
        await ledger.close()
        reopened = await SqliteLedger.open(db_path)
        await reopened.close()
    finally:
        await inspection.close()


async def test_task_store_persists_lossless_protobuf_and_scopes_crud(
    ledger: SqliteLedger, db_path: Path
) -> None:
    store = ledger.task_store_for("review")
    assert isinstance(store, VersionedTaskStore)
    other_agent = ledger.task_store_for("prepare")
    alice, bob = caller(), caller("bob")

    original = make_task(timestamp_ns=123_456_789)
    original.history.append(
        Message(
            message_id="request",
            role=Role.ROLE_USER,
            parts=[Part(text="Review the sample.", media_type="text/plain")],
            metadata={"tags": ["one", "two"]},
        )
    )
    original.artifacts.append(
        Artifact(
            artifact_id="artifact",
            parts=[
                Part(raw=b"\x00\xff", filename="binary", media_type="application/octet-stream"),
                new_data_part({"numbers": [1, 2], "ok": True}, media_type="application/json"),
                Part(url="https://example.invalid/object", media_type="image/png"),
            ],
        )
    )
    original.metadata.update({"nested": {"present": True}})
    # Unknown protobuf field 100, wire type varint, value 42.
    original.MergeFromString(b"\xa0\x06\x2a")
    expected = original.SerializeToString(deterministic=True)

    version = await store.save(original, prev_version=TaskVersion.MISSING, context=alice)
    original.ClearField("artifacts")
    assert await store.get(original.id, bob) is None
    assert await other_agent.get(original.id, alice) is None
    await store.delete(original.id, bob)
    await other_agent.delete(original.id, alice)
    assert (await store.list(ListTasksRequest(), bob)).total_size == 0
    assert (await other_agent.list(ListTasksRequest(), alice)).total_size == 0

    await store.save(make_task(context_id="bob"), prev_version=TaskVersion.MISSING, context=bob)
    await other_agent.save(
        make_task(context_id="prepare"), prev_version=TaskVersion.MISSING, context=alice
    )
    await ledger.close()
    reopened = await SqliteLedger.open(db_path)
    try:
        store = reopened.task_store_for("review")
        stored = await store.get(original.id, alice)
        assert stored is not None and stored.version == version
        assert stored.task.SerializeToString(deterministic=True) == expected
        page = await store.list(ListTasksRequest(), alice)
        assert len(page.tasks) == page.total_size == 1
        assert page.tasks[0].SerializeToString(deterministic=True) == expected
        stored.task.ClearField("history")
        assert (await store.get(original.id, alice)).task.history

        await store.delete(original.id, alice)
        assert await store.get(original.id, alice) is None
        assert (await store.get(original.id, bob)).task.context_id == "bob"
        isolated = reopened.task_store_for("prepare")
        assert (await isolated.get(original.id, alice)).task.context_id == "prepare"
    finally:
        await reopened.close()


async def test_list_uses_a2a_filters_and_stable_page_tokens(ledger: SqliteLedger) -> None:
    context = ServerCallContext(state={"principal": "alice"})
    store = ledger.task_store_for("review", owner_resolver=lambda call: call.state["principal"])
    for task in (
        make_task("a", timestamp_ns=100),
        make_task("b", timestamp_ns=100),
        make_task("c", timestamp_ns=99),
        make_task("missing-timestamp"),
        make_task("other-context", context_id="other", timestamp_ns=101),
        make_task("completed", state=COMPLETED, timestamp_ns=102),
    ):
        await store.save(task, prev_version=TaskVersion.MISSING, context=context)

    params = ListTasksRequest(context_id="context", status=WORKING, page_size=1)
    page = await store.list(params, context)
    assert (page.total_size, page.page_size, [t.id for t in page.tasks]) == (4, 1, ["b"])
    assert page.next_page_token
    await store.delete("b", context)
    params.page_token = page.next_page_token
    next_page = await store.list(params, context)
    assert (next_page.total_size, [t.id for t in next_page.tasks]) == (3, ["a"])

    params.page_token = encode_page_token("c")
    assert [t.id for t in (await store.list(params, context)).tasks] == ["c"]
    params.page_token = encode_page_token("other-context")
    with pytest.raises(InvalidParamsError, match="page token"):
        await store.list(params, context)
    params.ClearField("page_token")
    params.ClearField("page_size")
    params.status_timestamp_after.FromNanoseconds(100)
    assert [t.id for t in (await store.list(params, context)).tasks] == ["a"]


async def test_idempotency_reservations_are_atomic_and_scoped(
    ledger: SqliteLedger, db_path: Path
) -> None:
    other = await SqliteLedger.open(db_path, exclusive=False)
    try:
        results = await asyncio.gather(
            ledger.reserve_message("alice", "review", "message", "same", "first"),
            other.reserve_message("alice", "review", "message", "same", "second"),
        )
        assert sum(is_new for _, is_new in results) == 1
        task_id = results[0][0]
        assert {result[0] for result in results} == {task_id}
        assert await other.reserve_message("alice", "review", "message", "same", "retry") == (
            task_id,
            False,
        )
        with pytest.raises(DuplicateMessageConflictError):
            await other.reserve_message("alice", "review", "message", "different", "conflict")
        assert await other.reserve_message("bob", "review", "message", "different", "bob") == (
            "bob",
            True,
        )
        assert await other.reserve_message(
            "alice", "prepare", "message", "different", "prepare"
        ) == (
            "prepare",
            True,
        )
    finally:
        await other.close()


async def test_run_lineage_monotonic_events_and_standalone_finish(
    ledger: SqliteLedger, db_path: Path
) -> None:
    await ledger.start_run(
        "parent", owner="alice", agent_name="prepare", task_id="parent-task", context_id="context"
    )
    await ledger.start_run(
        "child",
        owner="alice",
        agent_name="review",
        task_id="child-task",
        context_id="context",
        parent_run_id="parent",
        workspace_root_id="workspace",
        workspace_access="read_only",
    )
    store = ledger.task_store_for("review")
    await store.save(make_task("child-task"), prev_version=TaskVersion.MISSING, context=caller())
    with closing(sqlite3.connect(db_path)) as db:
        assert db.execute("PRAGMA journal_mode").fetchone() == ("wal",)
        assert db.execute("SELECT parent_agent, parent_task_id FROM a2a_tasks").fetchone() == (
            "prepare",
            "parent-task",
        )

    started = StartedEvent(run_id="child", seq=0, native_session_ref="native-session")
    token = TextEvent(run_id="child", seq=2, text="report")
    await ledger.append_event(started)
    await ledger.append_event(token)
    for seq in (1, 2):
        with pytest.raises(ValueError, match="strictly increase"):
            await ledger.append_event(TextEvent(run_id="child", seq=seq, text="old"))
    with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
        await ledger.start_run(
            "orphan",
            owner="alice",
            agent_name="review",
            task_id="orphan",
            context_id="context",
            parent_run_id="missing",
        )

    finished = FinishedEvent(run_id="child", seq=3, status="completed")
    with pytest.raises(ValueError, match="FinishedEvent"):
        await ledger.append_event(finished)
    await ledger.finish_run(
        "child", status="completed", finished_event=finished, usage={"tokens": 3}
    )
    with pytest.raises(ValueError, match="dispatched"):
        await ledger.append_event(TextEvent(run_id="child", seq=4, text="late"))

    child = await ledger.get_run("child")
    assert child is not None and child["status"] == "completed"
    assert child["workspace_root_id"] == "workspace" and child["workspace_access"] == "read_only"
    assert child["native_session_ref"] == "native-session"
    assert child["usage"] == {"tokens": 3} and child["ended_at"] == finished.timestamp
    assert await ledger.get_run_events("child") == [started, token, finished]
    assert await ledger.list_runs(parent_run_id="parent", task_id="child-task") == [child]
    assert await ledger.list_runs(parent_run_id="parent", task_id="parent-task") == []
    assert (await store.get("child-task", caller())).task.status.state == WORKING


async def test_terminal_commit_rolls_back_together_and_can_retry(
    ledger: SqliteLedger, db_path: Path
) -> None:
    context = caller()
    store = ledger.task_store_for("review")
    working = make_task()
    version = await store.save(working, prev_version=TaskVersion.MISSING, context=context)
    await ledger.start_run(
        "run", owner="alice", agent_name="review", task_id=working.id, context_id=working.context_id
    )
    await ledger.append_event(StartedEvent(run_id="run", seq=0))
    finished = FinishedEvent(run_id="run", seq=1, status="completed", reason="done")
    artifact = Artifact(artifact_id="result", parts=[Part(text="ok")])
    await ledger.stage_terminal_run(
        working.id,
        owner="alice",
        agent_name="review",
        run_id="run",
        status="completed",
        reason="done",
        native_session_ref="native",
        usage={"tokens": 5},
        finished_event=finished,
        artifacts=[artifact],
    )

    updated = make_task()
    updated.metadata.update({"in_progress": True})
    current = await store.save(updated, prev=working, prev_version=version, context=context)
    assert (await ledger.get_run("run"))["status"] == "dispatched"
    with pytest.raises(ConcurrentTaskModificationError):
        await store.save(make_task(state=CANCELED), prev_version=current, context=context)

    final = make_task(state=COMPLETED)
    with pytest.raises(ConcurrentTaskModificationError):
        await store.save(final, prev=working, prev_version=version, context=context)

    with closing(sqlite3.connect(db_path)) as db:
        db.execute(
            """
            CREATE TRIGGER fail_finished BEFORE INSERT ON run_events WHEN NEW.kind = 'finished'
            BEGIN SELECT RAISE(ABORT, 'injected event failure'); END
            """
        )
        with pytest.raises(sqlite3.IntegrityError, match="injected event failure"):
            await store.save(final, prev=updated, prev_version=current, context=context)
        stored = await store.get(final.id, context)
        assert stored is not None and (stored.task, stored.version) == (updated, current)
        assert (await ledger.get_run("run"))["status"] == "dispatched"
        assert [event.kind for event in await ledger.get_run_events("run")] == ["started"]
        db.execute("DROP TRIGGER fail_finished")

    committed = await store.save(final, prev=updated, prev_version=current, context=context)
    run = await ledger.get_run("run")
    assert run is not None and run["status"] == "completed" and run["reason"] == "done"
    assert run["usage"] == {"tokens": 5} and run["native_session_ref"] == "native"
    assert (await ledger.get_run_events("run"))[-1] == finished
    assert list(final.artifacts) == [artifact]
    for stale_version in (current, committed):
        with pytest.raises(ConcurrentTaskModificationError):
            await store.save(updated, prev_version=stale_version, context=context)
    stored = await store.get(final.id, context)
    assert stored is not None and (stored.task, stored.version) == (final, committed)


async def test_pause_and_checkpoint_claim_each_commit_atomically(
    ledger: SqliteLedger, db_path: Path
) -> None:
    context = caller()
    scope = {"owner": "alice", "agent_name": "review", "task_id": "task", "context_id": "context"}
    store = ledger.task_store_for("review")
    version = await store.save(make_task(), prev_version=TaskVersion.MISSING, context=context)
    await ledger.start_run("pause", **scope)
    event = FinishedEvent(run_id="pause", seq=0, status="incomplete", reason="input_required")
    await ledger.stage_paused_run(
        **scope,
        run_id="pause",
        question="Which cutoff?",
        continuation={"sample": 1},
        workspace_req=None,
        schema_def=None,
        native_session_ref="paused-session",
        finished_event=event,
    )
    paused = make_task(state=INPUT_REQUIRED)
    with closing(sqlite3.connect(db_path)) as db:
        db.execute(
            """
            CREATE TRIGGER fail_pause BEFORE INSERT ON run_events WHEN NEW.kind = 'finished'
            BEGIN SELECT RAISE(ABORT, 'injected pause failure'); END
            """
        )
        with pytest.raises(sqlite3.IntegrityError, match="injected pause failure"):
            await store.save(paused, prev_version=version, context=context)
        assert await ledger.peek_continuation(**scope) is None
        assert await ledger.get_run_events("pause") == []
        assert (await ledger.get_run("pause"))["status"] == "dispatched"
        assert (await store.get("task", context)).task.status.state == WORKING
        db.execute("DROP TRIGGER fail_pause")

    await store.save(paused, prev_version=version, context=context)
    assert (await store.get("task", context)).task == paused
    checkpoint = await ledger.peek_continuation(**scope)
    assert checkpoint is not None and checkpoint["continuation"] == {"sample": 1}
    assert await ledger.get_run_events("pause") == [event]
    assert (await ledger.get_run("pause"))["status"] == "incomplete"

    with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
        await ledger.start_run(
            "reply", **scope, parent_run_id="unknown", continuation_run_id="pause"
        )
    assert await ledger.peek_continuation(**scope) == checkpoint
    assert await ledger.get_run("reply") is None
    await ledger.start_run("reply", **scope, parent_run_id="pause", continuation_run_id="pause")
    assert await ledger.peek_continuation(**scope) is None
    assert (await ledger.get_run("reply"))["status"] == "dispatched"


async def test_sdk_cancel_overrides_a_stale_version_only_for_a_live_task(
    ledger: SqliteLedger, db_path: Path
) -> None:
    context = caller()
    store = ledger.task_store_for("review")
    version = await store.save(make_task(), prev_version=TaskVersion.MISSING, context=context)
    other = await SqliteLedger.open(db_path, exclusive=False)
    try:
        current = await other.task_store_for("review").save(
            make_task(state=INPUT_REQUIRED), prev_version=version, context=context
        )
        canceled = make_task(state=CANCELED)
        newest = await store.save(canceled, prev_version=version, context=context)
        assert newest.is_after(current)
        assert (await store.get("task", context)).task == canceled
        with pytest.raises(ConcurrentTaskModificationError):
            await store.save(canceled, prev_version=newest, context=context)
        with pytest.raises(ConcurrentTaskModificationError):
            await store.save(
                make_task("absent", state=CANCELED),
                prev_version=TaskVersion.MISSING,
                context=context,
            )
    finally:
        await other.close()


async def test_continuation_survives_reopen_and_is_claimed_once(
    ledger: SqliteLedger, db_path: Path
) -> None:
    context = caller()
    await ledger.start_run(
        "paused-run", owner="alice", agent_name="review", task_id="task", context_id="context"
    )
    status = {
        "context_id": "context",
        "run_id": "paused-run",
        "question": "Which cutoff?",
        "continuation": {"rows": [1, 2]},
        "workspace_req": WorkspaceRequest(root="workspace"),
        "schema_def": {"type": "object"},
        "native_session_ref": "native",
    }
    await ledger.save_continuation("alice", "review", "task", **status)
    await ledger.task_store_for("review").save(
        make_task(state=INPUT_REQUIRED), prev_version=TaskVersion.MISSING, context=context
    )
    await ledger.close()

    first = await SqliteLedger.open(db_path)
    second = await SqliteLedger.open(db_path, exclusive=False)
    try:
        assert await first.consume_continuation("alice", "review", "task", "wrong") is None
        assert await first.consume_continuation("bob", "review", "task", "context") is None
        assert await first.consume_continuation("alice", "prepare", "task", "context") is None
        claims = await asyncio.gather(
            first.consume_continuation("alice", "review", "task", "context"),
            second.consume_continuation("alice", "review", "task", "context"),
        )
        assert claims.count(None) == 1
        expected = dict(status, workspace_req=status["workspace_req"].model_dump(mode="json"))
        assert next(claim for claim in claims if claim is not None) == expected
        await first.save_continuation("alice", "review", "task", **status)
        assert await second.consume_continuation("alice", "review", "task", "context") is None
        await first.save_continuation(
            "alice", "review", "task", **dict(status, question="Which month?")
        )
        await second.clear_continuation("alice", "review", "task")
        assert await first.consume_continuation("alice", "review", "task", "context") is None
    finally:
        await second.close()
        await first.close()


async def test_crash_recovery_fails_active_tasks_and_preserves_paused_tasks(
    ledger: SqliteLedger, db_path: Path
) -> None:
    context = caller()
    store = ledger.task_store_for("review")
    active = make_task("active", timestamp_ns=100)
    active.artifacts.add(artifact_id="partial").parts.add(text="Unconfirmed result")
    active.metadata.update({"caller_metadata": "retained", LIFECYCLE_EXT_URI: {"other": True}})
    version = await store.save(active, prev_version=TaskVersion.MISSING, context=context)
    paused = make_task("paused", state=INPUT_REQUIRED)
    paused_version = await store.save(paused, prev_version=TaskVersion.MISSING, context=context)
    terminal = make_task("terminal", state=COMPLETED)
    await store.save(terminal, prev_version=TaskVersion.MISSING, context=context)
    for state in (WORKING, SUBMITTED):
        abandoned = make_task(f"abandoned-{state}", state=state)
        abandoned.artifacts.add(artifact_id="partial").parts.add(text="Unconfirmed result")
        await store.save(abandoned, prev_version=TaskVersion.MISSING, context=context)
    for task_id in ("active", "paused", "terminal", "without-task"):
        await ledger.start_run(
            f"{task_id}-run",
            owner="alice",
            agent_name="review",
            task_id=task_id,
            context_id="context",
        )
    await ledger.start_run(
        "other-agent-run",
        owner="alice",
        agent_name="prepare",
        task_id="active",
        context_id="context",
    )
    started = StartedEvent(run_id="active-run", seq=5, native_session_ref="native-active")
    await ledger.append_event(started)
    for task_id in ("paused", "active"):
        await ledger.save_continuation(
            "alice",
            "review",
            task_id,
            context_id="context",
            run_id=f"{task_id}-run",
            question="Which month?",
            continuation={"month": None},
            workspace_req=None,
            schema_def=None,
            native_session_ref=None,
        )
    interrupted = make_task("interrupted-pause")
    await store.save(interrupted, prev_version=TaskVersion.MISSING, context=context)
    await ledger.start_run(
        "interrupted-run",
        owner="alice",
        agent_name="review",
        task_id=interrupted.id,
        context_id=interrupted.context_id,
    )
    await ledger.save_continuation(
        "alice",
        "review",
        interrupted.id,
        context_id=interrupted.context_id,
        run_id="interrupted-run",
        question="Which cutoff?",
        continuation={"sample": 1},
        workspace_req=None,
        schema_def=None,
        native_session_ref=None,
    )
    await ledger.finish_run(
        "interrupted-run",
        status="incomplete",
        reason="input_required",
        finished_event=FinishedEvent(
            run_id="interrupted-run", seq=0, status="incomplete", reason="input_required"
        ),
    )
    await ledger.close()

    recovered = await SqliteLedger.open(db_path)
    try:
        assert await recovered.recover_crashed_runs("review") == [
            "active-run",
            "terminal-run",
            "without-task-run",
        ]
        run = await recovered.get_run("active-run")
        assert run is not None and run["status"] == "indeterminate"
        assert run["native_session_ref"] == "native-active"
        assert run["reason"] == "recovered_after_crash"
        history = await recovered.get_run_events("active-run")
        assert history == [
            started,
            FinishedEvent(
                run_id="active-run",
                seq=6,
                status="indeterminate",
                reason="recovered_after_crash",
                timestamp=run["ended_at"],
            ),
        ]

        store = recovered.task_store_for("review")
        failed = await store.get("active", context)
        assert failed is not None and failed.version.is_after(version)
        assert failed.task.status.state == FAILED
        assert not failed.task.artifacts
        assert MessageToDict(failed.task.metadata) == {
            "caller_metadata": "retained",
            LIFECYCLE_EXT_URI: {"other": True, "indeterminate": True},
        }
        message = failed.task.status.message
        assert message.role == Role.ROLE_AGENT
        assert message.task_id == "active" and message.context_id == "context"
        assert "outcome is unknown" in message.parts[0].text
        with pytest.raises(ConcurrentTaskModificationError):
            await store.save(active, prev_version=version, context=context)

        actual_paused = await store.get("paused", context)
        assert actual_paused is not None
        assert (actual_paused.task, actual_paused.version) == (paused, paused_version)
        assert (await recovered.get_run("paused-run"))["status"] == "dispatched"
        assert await recovered.get_run_events("paused-run") == []
        restored = (await store.get(interrupted.id, context)).task
        assert restored.status.state == INPUT_REQUIRED
        assert restored.status.message.parts[0].text == "Which cutoff?"
        assert await recovered.peek_continuation(
            "alice", "review", restored.id, restored.context_id
        )
        assert (await store.get("terminal", context)).task == terminal
        assert await store.get("without-task", context) is None
        for state in (WORKING, SUBMITTED):
            recovered_task = (await store.get(f"abandoned-{state}", context)).task
            assert recovered_task.status.state == FAILED and not recovered_task.artifacts
            assert MessageToDict(recovered_task.metadata) == {
                LIFECYCLE_EXT_URI: {"indeterminate": True}
            }
        assert (await recovered.get_run("other-agent-run"))["status"] == "dispatched"
        assert await recovered.consume_continuation("alice", "review", "active", "context") is None
        assert await recovered.recover_crashed_runs("review") == []
        assert await recovered.consume_continuation("alice", "review", "paused", "context")
        assert await recovered.recover_crashed_runs() == ["paused-run", "other-agent-run"]
        assert (await store.get("paused", context)).task.status.state == FAILED
    finally:
        await recovered.close()


class JournalExecutor(AgentExecutor):
    def __init__(self, ledger: SqliteLedger):
        self.ledger = ledger

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        assert context.task_id and context.context_id
        await self.ledger.start_run(
            "native-run",
            owner="alice",
            agent_name="review",
            task_id=context.task_id,
            context_id=context.context_id,
        )
        task = make_task(
            context.task_id,
            context_id=context.context_id,
            state=TaskState.TASK_STATE_SUBMITTED,
        )
        await event_queue.enqueue_event(task)
        updater = TaskUpdater(event_queue, task.id, task.context_id)
        await updater.start_work()
        await self.ledger.append_event(StartedEvent(run_id="native-run", seq=0))
        await updater.add_artifact([Part(text="ok")], artifact_id="result")
        await self.ledger.stage_terminal_run(
            task.id,
            owner="alice",
            agent_name="review",
            run_id="native-run",
            status="completed",
            reason="",
            native_session_ref=None,
            usage=None,
            finished_event=FinishedEvent(run_id="native-run", seq=1, status="completed"),
        )
        await updater.complete()

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise AssertionError("The test executor should complete without cancellation")


async def test_v2_publishes_terminal_state_after_ledger_commit(
    ledger: SqliteLedger, db_path: Path
) -> None:
    assert DefaultRequestHandler is DefaultRequestHandlerV2
    card = AgentCard(
        name="Review",
        description="Test agent",
        version="1.0",
        supported_interfaces=[
            AgentInterface(
                url="https://example.invalid", protocol_binding="JSONRPC", protocol_version="1.0"
            )
        ],
        capabilities=AgentCapabilities(streaming=True),
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        skills=[AgentSkill(id="review", name="Review", description="Review text", tags=["test"])],
    )
    with pytest.warns(UserWarning, match="cross-replica streaming is disabled"):
        handler = DefaultRequestHandler(
            agent_executor=JournalExecutor(ledger),
            task_store=ledger.task_store_for("review"),
            agent_card=card,
        )
    observer = await SqliteLedger.open(db_path, exclusive=False)
    context = caller()
    try:
        request = SendMessageRequest(
            message=Message(message_id="send", role=Role.ROLE_USER, parts=[Part(text="Review")])
        )
        task_id = None
        async for event in handler.on_message_send_stream(request, context):
            if isinstance(event, TaskStatusUpdateEvent) and event.status.state == COMPLETED:
                task_id = event.task_id
                observed = await observer.task_store_for("review").get(task_id, context)
                assert observed is not None and observed.task.status.state == COMPLETED
                assert observed.task.artifacts[0].parts[0].text == "ok"
                assert (await observer.get_run("native-run"))["status"] == "completed"
                finished = (await observer.get_run_events("native-run"))[-1]
                assert isinstance(finished, FinishedEvent) and finished.seq == 1
        assert task_id is not None
        fetched = await handler.on_get_task(GetTaskRequest(id=task_id), context)
        assert fetched is not None and fetched.status.state == COMPLETED
        result = await handler.on_list_tasks(ListTasksRequest(include_artifacts=True), context)
        assert list(result.tasks) == [fetched]
        with pytest.raises(TaskNotFoundError):
            await handler.on_get_task(GetTaskRequest(id=task_id), caller("bob"))
    finally:
        await handler.aclose()
        await observer.close()
