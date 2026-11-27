"""Durable A2A task projections and native run records in one SQLite database."""

from __future__ import annotations

import asyncio
import fcntl
import json
import os
import sqlite3
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self
from uuid import uuid4

from a2a.server.cluster.task_store import (
    ConcurrentTaskModificationError,
    StoredTask,
    VersionedTaskStore,
)
from a2a.server.cluster.version import TaskVersion
from a2a.server.context import ServerCallContext
from a2a.server.events import Event
from a2a.server.owner_resolver import OwnerResolver, resolve_user_scope
from a2a.types import (
    Artifact,
    ListTasksRequest,
    ListTasksResponse,
    Message,
    Part,
    Role,
    Task,
    TaskState,
)
from a2a.utils.constants import DEFAULT_LIST_TASKS_PAGE_SIZE, TERMINAL_TASK_STATES
from a2a.utils.errors import InvalidParamsError
from a2a.utils.task import (
    ListTasksCursor,
    decode_list_tasks_cursor,
    decode_page_token,
    encode_list_tasks_cursor,
    validate_page_size,
)
from pydantic import TypeAdapter

from john.types import (
    LIFECYCLE_EXT_URI,
    BusyError,
    DuplicateMessageConflictError,
    FinishedEvent,
    ResumeUnavailableError,
    RuntimeEvent,
    StartedEvent,
    TurnStatus,
    WorkspaceAccess,
    WorkspaceRequest,
    utc_now_iso,
)

_SCHEMA = """
BEGIN IMMEDIATE;
CREATE TABLE IF NOT EXISTS a2a_tasks (
    owner TEXT NOT NULL,
    agent_name TEXT NOT NULL,
    task_id TEXT NOT NULL,
    context_id TEXT NOT NULL,
    state INTEGER NOT NULL,
    version INTEGER NOT NULL,
    parent_agent TEXT,
    parent_task_id TEXT,
    task_proto BLOB NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (owner, agent_name, task_id)
);
CREATE TABLE IF NOT EXISTS message_reservations (
    owner TEXT NOT NULL,
    agent_name TEXT NOT NULL,
    message_id TEXT NOT NULL,
    fingerprint TEXT NOT NULL,
    task_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (owner, agent_name, message_id)
);
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT NOT NULL PRIMARY KEY,
    owner TEXT NOT NULL,
    agent_name TEXT NOT NULL,
    task_id TEXT NOT NULL,
    context_id TEXT NOT NULL,
    parent_run_id TEXT REFERENCES runs (run_id) CHECK (parent_run_id <> run_id),
    workspace_root_id TEXT,
    workspace_access TEXT,
    native_session_ref TEXT,
    status TEXT NOT NULL,
    reason TEXT,
    usage_json TEXT,
    started_at TEXT NOT NULL,
    ended_at TEXT
);
CREATE TABLE IF NOT EXISTS run_events (
    run_id TEXT NOT NULL REFERENCES runs (run_id),
    seq INTEGER NOT NULL CHECK (seq >= 0),
    kind TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (run_id, seq)
);
CREATE TABLE IF NOT EXISTS continuations (
    owner TEXT NOT NULL,
    agent_name TEXT NOT NULL,
    task_id TEXT NOT NULL,
    context_id TEXT NOT NULL,
    run_id TEXT NOT NULL REFERENCES runs (run_id),
    question TEXT NOT NULL,
    continuation_json TEXT NOT NULL,
    workspace_json TEXT,
    schema_json TEXT,
    native_session_ref TEXT,
    consumed INTEGER NOT NULL DEFAULT 0 CHECK (consumed IN (0, 1)),
    created_at TEXT NOT NULL,
    PRIMARY KEY (owner, agent_name, task_id)
);
CREATE INDEX IF NOT EXISTS tasks_by_context
    ON a2a_tasks (owner, agent_name, context_id, updated_at);
CREATE INDEX IF NOT EXISTS runs_by_task ON runs (task_id);
CREATE INDEX IF NOT EXISTS runs_by_parent ON runs (parent_run_id);
COMMIT;
"""

_TASK_SCOPE = "owner = ? AND agent_name = ? AND task_id = ?"
_EVENT_TYPE = TypeAdapter(RuntimeEvent)
_RECOVERY_REASON = "recovered_after_crash"
RUN_TERMINAL_STATES: dict[TurnStatus, TaskState.ValueType] = {
    "completed": TaskState.TASK_STATE_COMPLETED,
    "cancelled": TaskState.TASK_STATE_CANCELED,
    "failed": TaskState.TASK_STATE_FAILED,
    "incomplete": TaskState.TASK_STATE_FAILED,
    "indeterminate": TaskState.TASK_STATE_FAILED,
}

type _TaskKey = tuple[str, str, str]
type _JsonObject = dict[str, Any]
type _Checkpoint = tuple[str, str, str, str, str | None, str | None, str | None]


@dataclass(frozen=True)
class _TerminalRun:
    event: FinishedEvent
    reason: str
    native_session_ref: str | None
    usage_json: str | None


@dataclass(frozen=True)
class _StagedRun:
    state: TaskState.ValueType
    terminal: _TerminalRun
    checkpoint: _Checkpoint | None = None
    artifacts: tuple[bytes, ...] = ()


def _encode_json(value: _JsonObject | None) -> str | None:
    return None if value is None else json.dumps(value, sort_keys=True, allow_nan=False)


def _decode_json(value: str | None) -> _JsonObject | None:
    return None if value is None else json.loads(value)


def _terminal_run(
    run_id: str,
    status: TurnStatus,
    reason: str,
    native_session_ref: str | None,
    usage: _JsonObject | None,
    finished_event: FinishedEvent,
) -> _TerminalRun:
    if finished_event.run_id != run_id or finished_event.status != status:
        raise ValueError("FinishedEvent must match the run ID and terminal status")
    return _TerminalRun(finished_event, reason, native_session_ref, _encode_json(usage))


def _task_row(db: sqlite3.Connection, key: _TaskKey) -> sqlite3.Row | None:
    return db.execute(f"SELECT * FROM a2a_tasks WHERE {_TASK_SCOPE}", key).fetchone()


def _write_task(db: sqlite3.Connection, key: _TaskKey, task: Task, version: int) -> None:
    now = utc_now_iso()
    db.execute(
        """
        INSERT INTO a2a_tasks (
            owner, agent_name, task_id, context_id, state, version, task_proto,
            created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (owner, agent_name, task_id) DO UPDATE SET
            context_id = excluded.context_id, state = excluded.state, version = excluded.version,
            task_proto = excluded.task_proto, updated_at = excluded.updated_at
        """,
        (*key, task.context_id, task.status.state, version, task.SerializeToString(), now, now),
    )


def _set_agent_status(task: Task, state: TaskState.ValueType, text: str) -> None:
    task.status.state = state
    task.status.timestamp.GetCurrentTime()
    task.status.message.CopyFrom(
        Message(
            message_id=str(uuid4()),
            task_id=task.id,
            context_id=task.context_id,
            role=Role.ROLE_AGENT,
            parts=[Part(text=text, media_type="text/plain")],
        )
    )


def _link_parent_task(db: sqlite3.Connection, key: _TaskKey) -> None:
    parent = db.execute(
        """
        SELECT p.agent_name, p.task_id
        FROM runs AS r JOIN runs AS p ON r.parent_run_id = p.run_id
        WHERE (r.owner, r.agent_name, r.task_id) = (?, ?, ?)
          AND (p.owner, p.agent_name, p.task_id) <> (r.owner, r.agent_name, r.task_id)
        ORDER BY r.started_at, r.run_id LIMIT 1
        """,
        key,
    ).fetchone()
    if parent is not None:
        db.execute(
            f"""
            UPDATE a2a_tasks SET parent_agent = ?, parent_task_id = ?
            WHERE {_TASK_SCOPE} AND parent_task_id IS NULL
            """,
            (*parent, *key),
        )


def _run_key(row: sqlite3.Row) -> _TaskKey:
    return row["owner"], row["agent_name"], row["task_id"]


def _dispatched_run(db: sqlite3.Connection, run_id: str) -> sqlite3.Row:
    row = db.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
    if row is None or row["status"] != "dispatched":
        raise ValueError("Run must exist and be dispatched")
    return row


def _next_seq(db: sqlite3.Connection, run_id: str) -> int:
    return db.execute(
        "SELECT COALESCE(MAX(seq), -1) + 1 FROM run_events WHERE run_id = ?", (run_id,)
    ).fetchone()[0]


def _insert_event(db: sqlite3.Connection, event: RuntimeEvent) -> None:
    if event.seq < _next_seq(db, event.run_id):
        raise ValueError("Run event sequence numbers must strictly increase")
    db.execute(
        """
        INSERT INTO run_events (run_id, seq, kind, payload_json, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (event.run_id, event.seq, event.kind, event.model_dump_json(), event.timestamp),
    )


def _commit_terminal_run(db: sqlite3.Connection, terminal: _TerminalRun) -> None:
    event = terminal.event
    db.execute(
        """
        UPDATE runs SET status = ?, reason = ?, usage_json = ?, ended_at = ?,
                        native_session_ref = COALESCE(?, native_session_ref)
        WHERE run_id = ?
        """,
        (
            event.status,
            terminal.reason,
            terminal.usage_json,
            event.timestamp,
            terminal.native_session_ref,
            event.run_id,
        ),
    )
    _insert_event(db, event)


def _read_run(row: sqlite3.Row) -> _JsonObject:
    result = dict(row)
    result["usage"] = _decode_json(result.pop("usage_json"))
    return result


def _reservation_id(
    db: sqlite3.Connection, owner: str, agent_name: str, message_id: str, fingerprint: str
) -> str | None:
    row = db.execute(
        """
        SELECT fingerprint, task_id FROM message_reservations
        WHERE owner = ? AND agent_name = ? AND message_id = ?
        """,
        (owner, agent_name, message_id),
    ).fetchone()
    if row is None:
        return None
    if row["fingerprint"] != fingerprint:
        raise DuplicateMessageConflictError("Message ID has a different fingerprint")
    return row["task_id"]


def _encode_checkpoint(
    context_id: str,
    run_id: str,
    question: str,
    continuation: _JsonObject,
    workspace_req: WorkspaceRequest | _JsonObject | None,
    schema_def: _JsonObject | None,
    native_session_ref: str | None,
) -> _Checkpoint:
    if isinstance(workspace_req, WorkspaceRequest):
        workspace_req = workspace_req.model_dump(mode="json")
    return (
        context_id,
        run_id,
        question,
        json.dumps(continuation, sort_keys=True, allow_nan=False),
        _encode_json(workspace_req),
        _encode_json(schema_def),
        native_session_ref,
    )


def _write_checkpoint(db: sqlite3.Connection, key: _TaskKey, payload: _Checkpoint) -> None:
    db.execute(
        """
        INSERT INTO continuations (
            owner, agent_name, task_id, context_id, run_id, question, continuation_json,
            workspace_json, schema_json, native_session_ref, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (owner, agent_name, task_id) DO UPDATE SET
            context_id = excluded.context_id, run_id = excluded.run_id,
            question = excluded.question, continuation_json = excluded.continuation_json,
            workspace_json = excluded.workspace_json, schema_json = excluded.schema_json,
            native_session_ref = excluded.native_session_ref,
            consumed = 0, created_at = excluded.created_at
        WHERE (continuations.context_id, continuations.run_id, continuations.question,
               continuations.continuation_json, continuations.workspace_json,
               continuations.schema_json, continuations.native_session_ref)
            IS NOT (excluded.context_id, excluded.run_id, excluded.question,
                    excluded.continuation_json, excluded.workspace_json,
                    excluded.schema_json, excluded.native_session_ref)
        """,
        (*key, *payload, utc_now_iso()),
    )


def _read_checkpoint(db: sqlite3.Connection, key: _TaskKey, context_id: str) -> _JsonObject | None:
    row = db.execute(
        f"SELECT * FROM continuations WHERE {_TASK_SCOPE} AND context_id = ? AND consumed = 0",
        (*key, context_id),
    ).fetchone()
    if row is None:
        return None
    return {
        "context_id": row["context_id"],
        "run_id": row["run_id"],
        "question": row["question"],
        "continuation": _decode_json(row["continuation_json"]),
        "workspace_req": _decode_json(row["workspace_json"]),
        "schema_def": _decode_json(row["schema_json"]),
        "native_session_ref": row["native_session_ref"],
    }


class SqliteLedger:
    """Own a WAL connection and share it across agent-scoped task store views."""

    def __init__(self) -> None:
        self._connection: sqlite3.Connection | None = None
        self._lock_fd: int | None = None
        self._lock = asyncio.Lock()
        self._worker: asyncio.Task[Any] | None = None
        self._staged: dict[_TaskKey, _StagedRun] = {}

    @classmethod
    async def open(cls, db_path: Path, *, exclusive: bool = True) -> Self:
        """Open a durable ledger. Inspection connections can skip the execution lock."""
        ledger = cls()

        def initialize() -> None:
            path = db_path.resolve()
            if exclusive:
                fd = os.open(f"{path}.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    os.close(fd)
                    raise BusyError("Ledger is locked by another active coordinator") from None
                except BaseException:
                    os.close(fd)
                    raise
                ledger._lock_fd = fd

            db = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
            ledger._connection = db
            try:
                db.row_factory = sqlite3.Row
                db.execute("PRAGMA busy_timeout = 5000")
                if db.execute("PRAGMA journal_mode = WAL").fetchone()[0] != "wal":
                    raise RuntimeError("Ledger requires a SQLite database with WAL support")
                db.execute("PRAGMA synchronous = FULL")
                db.execute("PRAGMA foreign_keys = ON")
                db.executescript(_SCHEMA)
            except BaseException:
                db.close()
                ledger._connection = None
                raise

        try:
            await ledger._call(initialize)
        except BaseException:
            await ledger.close()
            raise
        return ledger

    async def close(self) -> None:
        """Close after any current operation completes."""

        def close_connection() -> None:
            try:
                if self._connection is not None:
                    self._connection.close()
            finally:
                self._connection = None
                self._staged.clear()
                if self._lock_fd is not None:
                    os.close(self._lock_fd)
                    self._lock_fd = None

        await self._call(close_connection)

    @property
    def exclusive(self) -> bool:
        """Whether this ledger holds the coordinator's execution lock."""
        return self._lock_fd is not None

    async def _call[T](self, operation: Callable[[], T]) -> T:
        await self._lock.acquire()
        worker = asyncio.create_task(asyncio.to_thread(operation))
        self._worker = worker

        def release(done: asyncio.Task[T]) -> None:
            self._worker = None
            self._lock.release()
            if not done.cancelled():
                done.exception()

        # Keep the lock until SQLite finishes, even if the awaiting request is canceled.
        worker.add_done_callback(release)
        return await asyncio.shield(worker)

    @property
    def _db(self) -> sqlite3.Connection:
        if self._connection is None:
            raise RuntimeError("Ledger is closed")
        return self._connection

    @contextmanager
    def _write(self) -> Iterator[sqlite3.Connection]:
        db = self._db
        db.execute("BEGIN IMMEDIATE")
        with db:
            yield db

    def task_store_for(
        self, agent_name: str, *, owner_resolver: OwnerResolver = resolve_user_scope
    ) -> AgentTaskStore:
        """Use the SDK's request-scoped principal resolver by default."""
        return AgentTaskStore(self, agent_name, owner_resolver)

    async def reserve_message(
        self, owner: str, agent_name: str, message_id: str, fingerprint: str, proposed_task_id: str
    ) -> tuple[str, bool]:
        """Reserve one task ID per scoped message and verify retries' fingerprints."""

        def reserve() -> tuple[str, bool]:
            with self._write() as db:
                key = (owner, agent_name, message_id)
                task_id = _reservation_id(db, *key, fingerprint)
                if task_id is not None:
                    return task_id, False
                db.execute(
                    """
                    INSERT INTO message_reservations
                        (owner, agent_name, message_id, fingerprint, task_id, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (*key, fingerprint, proposed_task_id, utc_now_iso()),
                )
                return proposed_task_id, True

        return await self._call(reserve)

    async def lookup_message_reservation(
        self, owner: str, agent_name: str, message_id: str, fingerprint: str
    ) -> str | None:
        return await self._call(
            lambda: _reservation_id(self._db, owner, agent_name, message_id, fingerprint)
        )

    async def is_unsent_task(self, owner: str, agent_name: str, task_id: str) -> bool:
        """Check whether an admission has neither an SDK snapshot nor dispatch intent."""
        key = (owner, agent_name, task_id)

        def check() -> bool:
            return (
                _task_row(self._db, key) is None
                and self._db.execute(
                    f"SELECT 1 FROM runs WHERE {_TASK_SCOPE} LIMIT 1", key
                ).fetchone()
                is None
            )

        return await self._call(check)

    async def start_run(
        self,
        run_id: str,
        *,
        owner: str,
        agent_name: str,
        task_id: str,
        context_id: str,
        parent_run_id: str | None = None,
        workspace_root_id: str | None = None,
        workspace_access: WorkspaceAccess | None = None,
        continuation_run_id: str | None = None,
    ) -> None:
        """Record dispatch intent before invoking the harness."""
        if not self.exclusive:
            raise BusyError("Dispatch requires exclusive ledger ownership")
        key = (owner, agent_name, task_id)

        def start() -> None:
            with self._write() as db:
                if continuation_run_id is not None:
                    claimed = db.execute(
                        f"""
                        UPDATE continuations SET consumed = 1
                        WHERE {_TASK_SCOPE} AND context_id = ? AND run_id = ? AND consumed = 0
                        """,
                        (*key, context_id, continuation_run_id),
                    )
                    if claimed.rowcount != 1:
                        raise ResumeUnavailableError("No active continuation for this task")
                db.execute(
                    """
                    INSERT INTO runs (
                        run_id, owner, agent_name, task_id, context_id, parent_run_id,
                        workspace_root_id, workspace_access, status, started_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'dispatched', ?)
                    """,
                    (
                        run_id,
                        *key,
                        context_id,
                        parent_run_id,
                        workspace_root_id,
                        workspace_access,
                        utc_now_iso(),
                    ),
                )
                if parent_run_id is not None:
                    _link_parent_task(db, key)

        await self._call(start)

    async def append_event(self, event: RuntimeEvent) -> None:
        """Append a non-terminal event to a dispatched run."""
        if isinstance(event, FinishedEvent):
            raise ValueError("FinishedEvent requires stage_terminal_run or finish_run")

        def append() -> None:
            with self._write() as db:
                _dispatched_run(db, event.run_id)
                _insert_event(db, event)
                if isinstance(event, StartedEvent) and event.native_session_ref is not None:
                    db.execute(
                        "UPDATE runs SET native_session_ref = ? WHERE run_id = ?",
                        (event.native_session_ref, event.run_id),
                    )

        await self._call(append)

    async def stage_terminal_run(
        self,
        task_id: str,
        *,
        owner: str,
        agent_name: str,
        run_id: str,
        status: TurnStatus,
        reason: str,
        native_session_ref: str | None,
        usage: _JsonObject | None,
        finished_event: FinishedEvent,
        artifacts: Sequence[Artifact] = (),
    ) -> None:
        """Hold a run outcome until the handler saves the terminal task."""
        key = (owner, agent_name, task_id)
        terminal = _terminal_run(run_id, status, reason, native_session_ref, usage, finished_event)
        staged = _StagedRun(
            RUN_TERMINAL_STATES[status],
            terminal,
            artifacts=tuple(artifact.SerializeToString() for artifact in artifacts),
        )
        await self._call(lambda: self._stage(key, staged))

    async def stage_paused_run(
        self,
        task_id: str,
        *,
        owner: str,
        agent_name: str,
        run_id: str,
        context_id: str,
        question: str,
        continuation: _JsonObject,
        workspace_req: WorkspaceRequest | _JsonObject | None,
        schema_def: _JsonObject | None,
        native_session_ref: str | None,
        finished_event: FinishedEvent,
    ) -> None:
        """Stage a checkpoint and its run outcome for the INPUT_REQUIRED save."""
        key = (owner, agent_name, task_id)
        terminal = _terminal_run(
            run_id, "incomplete", "input_required", native_session_ref, None, finished_event
        )
        checkpoint = _encode_checkpoint(
            context_id,
            run_id,
            question,
            continuation,
            workspace_req,
            schema_def,
            native_session_ref,
        )
        staged = _StagedRun(TaskState.TASK_STATE_INPUT_REQUIRED, terminal, checkpoint)
        await self._call(lambda: self._stage(key, staged))

    def _stage(self, key: _TaskKey, staged: _StagedRun) -> None:
        run_id = staged.terminal.event.run_id
        run = _dispatched_run(self._db, run_id)
        if _run_key(run) != key or (
            staged.checkpoint is not None and run["context_id"] != staged.checkpoint[0]
        ):
            raise ValueError("Run belongs to a different task scope or context")
        previous = self._staged.get(key)
        if previous is not None and previous.terminal.event.run_id != run_id:
            raise ValueError("A different run is already staged for this task")
        self._staged[key] = staged

    def _unstage(self, key: _TaskKey, run_id: str) -> None:
        staged = self._staged.get(key)
        if staged is not None and staged.terminal.event.run_id == run_id:
            del self._staged[key]

    async def finish_run(
        self,
        run_id: str,
        *,
        status: TurnStatus,
        finished_event: FinishedEvent,
        reason: str = "",
        native_session_ref: str | None = None,
        usage: _JsonObject | None = None,
        project_failure: bool = False,
    ) -> None:
        """Commit a native terminal outcome when no terminal task save follows."""
        terminal = _terminal_run(run_id, status, reason, native_session_ref, usage, finished_event)
        if project_failure and status != "indeterminate":
            raise ValueError("Publication failure must be recorded as indeterminate")

        def finish() -> None:
            with self._write() as db:
                if project_failure:
                    current = db.execute("SELECT status FROM runs WHERE run_id = ?", (run_id,))
                    if (row := current.fetchone()) is not None and row["status"] != "dispatched":
                        # Publication may have reached the consumer before the queue failed.
                        return
                run = _dispatched_run(db, run_id)
                _commit_terminal_run(db, terminal)
                if project_failure:
                    self._project_recovery(
                        db,
                        _run_key(run),
                        run["context_id"],
                        text="The run result could not be published. Its outcome is indeterminate.",
                    )
            self._unstage(_run_key(run), run_id)

        await self._call(finish)

    async def get_run(self, run_id: str) -> _JsonObject | None:
        """Return a local run record with decoded usage."""

        def get() -> _JsonObject | None:
            row = self._db.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
            return None if row is None else _read_run(row)

        return await self._call(get)

    async def list_runs(
        self, task_id: str | None = None, parent_run_id: str | None = None
    ) -> list[_JsonObject]:
        """List run records in dispatch order, optionally filtering by task or parent."""
        filters, values = [], []
        if task_id is not None:
            filters.append("task_id = ?")
            values.append(task_id)
        if parent_run_id is not None:
            filters.append("parent_run_id = ?")
            values.append(parent_run_id)
        where = " WHERE " + " AND ".join(filters) if filters else ""

        def list_rows() -> list[_JsonObject]:
            rows = self._db.execute(
                f"SELECT * FROM runs{where} ORDER BY started_at, run_id", values
            )
            return [_read_run(row) for row in rows]

        return await self._call(list_rows)

    async def get_run_events(self, run_id: str) -> list[RuntimeEvent]:
        """Replay canonical events in their stored sequence order."""

        def events() -> list[RuntimeEvent]:
            rows = self._db.execute(
                "SELECT payload_json FROM run_events WHERE run_id = ? ORDER BY seq", (run_id,)
            )
            return [_EVENT_TYPE.validate_json(row["payload_json"]) for row in rows]

        return await self._call(events)

    async def save_continuation(
        self,
        owner: str,
        agent_name: str,
        task_id: str,
        *,
        context_id: str,
        run_id: str,
        question: str,
        continuation: _JsonObject,
        workspace_req: WorkspaceRequest | _JsonObject | None,
        schema_def: _JsonObject | None,
        native_session_ref: str | None,
    ) -> None:
        """Persist the current task-level checkpoint, preserving identical consumed retries."""
        key = (owner, agent_name, task_id)
        payload = _encode_checkpoint(
            context_id,
            run_id,
            question,
            continuation,
            workspace_req,
            schema_def,
            native_session_ref,
        )

        def save() -> None:
            with self._write() as db:
                run = _dispatched_run(db, run_id)
                if _run_key(run) != key or run["context_id"] != context_id:
                    raise ValueError("Continuation must match the run's task scope and context")
                task = _task_row(db, key)
                if task is not None and (
                    task["context_id"] != context_id or task["state"] in TERMINAL_TASK_STATES
                ):
                    raise ValueError("Continuation requires a matching non-terminal task")
                _write_checkpoint(db, key, payload)

        await self._call(save)

    async def peek_continuation(
        self, owner: str, agent_name: str, task_id: str, context_id: str
    ) -> _JsonObject | None:
        """Read a scoped checkpoint for validation before a new run claims it."""
        key = (owner, agent_name, task_id)
        return await self._call(lambda: _read_checkpoint(self._db, key, context_id))

    async def consume_continuation(
        self, owner: str, agent_name: str, task_id: str, context_id: str
    ) -> _JsonObject | None:
        """Claim one matching checkpoint and return its decoded resume inputs."""
        key = (owner, agent_name, task_id)

        def consume() -> _JsonObject | None:
            with self._write() as db:
                result = _read_checkpoint(db, key, context_id)
                if result is not None:
                    db.execute(f"UPDATE continuations SET consumed = 1 WHERE {_TASK_SCOPE}", key)
                return result

        return await self._call(consume)

    async def clear_continuation(self, owner: str, agent_name: str, task_id: str) -> None:
        """Discard the checkpoint for a scoped task."""

        def clear() -> None:
            with self._write() as db:
                db.execute(
                    f"DELETE FROM continuations WHERE {_TASK_SCOPE}", (owner, agent_name, task_id)
                )

        await self._call(clear)

    async def recover_crashed_runs(self, agent_name: str | None = None) -> list[str]:
        """Mark uncertain dispatched runs at startup and preserve valid paused tasks."""
        if not self.exclusive:
            raise BusyError("Crash recovery requires exclusive ledger ownership")

        def recover() -> list[str]:
            with self._write() as db:
                # Older ledgers could commit a checkpoint and native pause before the task save.
                interrupted_pauses = db.execute(
                    """
                    SELECT t.*, c.question FROM a2a_tasks AS t
                    JOIN continuations AS c USING (owner, agent_name, task_id)
                    JOIN runs AS r ON r.run_id = c.run_id
                    WHERE t.state IN (?, ?) AND (? IS NULL OR t.agent_name = ?)
                      AND c.context_id = t.context_id AND c.consumed = 0
                      AND r.status = 'incomplete' AND r.reason = 'input_required'
                    """,
                    (
                        TaskState.TASK_STATE_WORKING,
                        TaskState.TASK_STATE_SUBMITTED,
                        agent_name,
                        agent_name,
                    ),
                ).fetchall()
                for row in interrupted_pauses:
                    task = Task.FromString(row["task_proto"])
                    task.ClearField("artifacts")
                    _set_agent_status(task, TaskState.TASK_STATE_INPUT_REQUIRED, row["question"])
                    _write_task(db, _run_key(row), task, row["version"] + 1)
                rows = db.execute(
                    """
                    SELECT r.* FROM runs AS r
                    WHERE r.status = 'dispatched' AND (? IS NULL OR r.agent_name = ?)
                      AND NOT EXISTS (
                        SELECT 1 FROM continuations AS c
                        JOIN a2a_tasks AS t USING (owner, agent_name, task_id)
                        WHERE (c.owner, c.agent_name, c.task_id, c.context_id)
                            = (r.owner, r.agent_name, r.task_id, r.context_id)
                          AND t.context_id = c.context_id AND t.state = ? AND c.consumed = 0
                      )
                    ORDER BY r.started_at, r.run_id
                    """,
                    (agent_name, agent_name, TaskState.TASK_STATE_INPUT_REQUIRED),
                ).fetchall()
                for run in rows:
                    run_id = run["run_id"]
                    event = FinishedEvent(
                        run_id=run_id,
                        seq=_next_seq(db, run_id),
                        status="indeterminate",
                        reason=_RECOVERY_REASON,
                    )
                    _commit_terminal_run(db, _TerminalRun(event, _RECOVERY_REASON, None, None))
                    self._project_recovery(db, _run_key(run), run["context_id"])
                abandoned = db.execute(
                    """
                    SELECT t.owner, t.agent_name, t.task_id, t.context_id FROM a2a_tasks AS t
                    WHERE t.state IN (?, ?) AND (? IS NULL OR t.agent_name = ?)
                      AND NOT EXISTS (
                        SELECT 1 FROM continuations AS c
                        WHERE (c.owner, c.agent_name, c.task_id, c.context_id)
                            = (t.owner, t.agent_name, t.task_id, t.context_id)
                          AND c.consumed = 0
                      )
                    """,
                    (
                        TaskState.TASK_STATE_WORKING,
                        TaskState.TASK_STATE_SUBMITTED,
                        agent_name,
                        agent_name,
                    ),
                ).fetchall()
                for task in abandoned:
                    self._project_recovery(db, _run_key(task), task["context_id"])
            for run in rows:
                self._unstage(_run_key(run), run["run_id"])
            return [run["run_id"] for run in rows]

        return await self._call(recover)

    @staticmethod
    def _project_recovery(
        db: sqlite3.Connection,
        key: _TaskKey,
        context_id: str,
        *,
        text: str = "The process stopped during execution. The run outcome is unknown.",
    ) -> None:
        row = _task_row(db, key)
        if row is None or row["context_id"] != context_id or row["state"] in TERMINAL_TASK_STATES:
            return
        task = Task.FromString(row["task_proto"])
        task.ClearField("artifacts")
        _set_agent_status(task, TaskState.TASK_STATE_FAILED, text)
        task.metadata.fields[LIFECYCLE_EXT_URI].struct_value.fields[
            "indeterminate"
        ].bool_value = True
        _write_task(db, key, task, row["version"] + 1)
        db.execute(
            f"DELETE FROM continuations WHERE {_TASK_SCOPE} AND context_id = ?", (*key, context_id)
        )

    def _save_task(self, key: _TaskKey, task: Task, prev_version: TaskVersion) -> TaskVersion:
        state = task.status.state
        terminal = state in TERMINAL_TASK_STATES
        boundary = terminal or state == TaskState.TASK_STATE_INPUT_REQUIRED
        staged = self._staged.get(key) if boundary else None
        projection = task
        with self._write() as db:
            row = _task_row(db, key)
            version = 0 if row is None else row["version"]
            if row is not None and row["state"] in TERMINAL_TASK_STATES:
                raise ConcurrentTaskModificationError(task.id)
            if state == TaskState.TASK_STATE_CANCELED:
                if row is None:
                    raise ConcurrentTaskModificationError(task.id)
            elif TaskVersion(version) != prev_version:
                raise ConcurrentTaskModificationError(task.id)
            if staged is not None:
                if state != staged.state:
                    raise ConcurrentTaskModificationError(task.id)
                run = _dispatched_run(db, staged.terminal.event.run_id)
                if run["context_id"] != task.context_id:
                    raise ValueError("Staged run belongs to a different task context")
                if staged.checkpoint is not None:
                    _write_checkpoint(db, key, staged.checkpoint)
                _commit_terminal_run(db, staged.terminal)
                if staged.artifacts:
                    projection = Task()
                    projection.CopyFrom(task)
                    projection.artifacts.extend(
                        Artifact.FromString(data) for data in staged.artifacts
                    )
            _write_task(db, key, projection, version + 1)
            if row is None:
                _link_parent_task(db, key)
            if terminal:
                db.execute(f"DELETE FROM continuations WHERE {_TASK_SCOPE}", key)
        if staged is not None:
            del self._staged[key]
        if projection is not task:
            # The SDK returns this same snapshot to callers after save completes.
            task.CopyFrom(projection)
        return TaskVersion(version + 1)

    def _delete_task(self, key: _TaskKey) -> None:
        with self._write() as db:
            db.execute(f"DELETE FROM continuations WHERE {_TASK_SCOPE}", key)
            db.execute(f"DELETE FROM a2a_tasks WHERE {_TASK_SCOPE}", key)
        self._staged.pop(key, None)


def _list_page(tasks: Iterator[Task], params: ListTasksRequest) -> ListTasksResponse:
    cutoff = (
        params.status_timestamp_after.ToNanoseconds()
        if params.HasField("status_timestamp_after")
        else None
    )
    entries: list[tuple[ListTasksCursor, Task]] = []
    for task in tasks:
        timestamp = (
            task.status.timestamp.ToNanoseconds() if task.status.HasField("timestamp") else None
        )
        if cutoff is None or (timestamp is not None and timestamp >= cutoff):
            entries.append((ListTasksCursor(timestamp, task.id), task))
    entries.sort(key=lambda entry: entry[0].sort_key(), reverse=True)

    start = 0
    if params.page_token:
        cursor = decode_list_tasks_cursor(params.page_token)
        if cursor is not None:
            sort_key = cursor.sort_key()
            start = next(
                (i for i, (position, _) in enumerate(entries) if position.sort_key() < sort_key),
                len(entries),
            )
        else:
            task_id = decode_page_token(params.page_token)
            start = next((i for i, (_, task) in enumerate(entries) if task.id == task_id), -1)
            if start == -1:
                raise InvalidParamsError("Invalid page token")

    page_size = params.page_size if params.HasField("page_size") else DEFAULT_LIST_TASKS_PAGE_SIZE
    validate_page_size(page_size)
    end = start + page_size
    return ListTasksResponse(
        tasks=[task for _, task in entries[start:end]],
        next_page_token=encode_list_tasks_cursor(entries[end - 1][0]) if end < len(entries) else "",
        total_size=len(entries),
        page_size=page_size,
    )


class AgentTaskStore(VersionedTaskStore):
    """The SDK's versioned task store for one agent and a resolved caller."""

    def __init__(
        self, ledger: SqliteLedger, agent_name: str, owner_resolver: OwnerResolver
    ) -> None:
        self._ledger = ledger
        self._agent_name = agent_name
        self._owner_resolver = owner_resolver

    async def save(
        self,
        task: Task,
        *,
        event: Event | None = None,
        prev: Task | None = None,
        prev_version: TaskVersion,
        context: ServerCallContext,
    ) -> TaskVersion:
        key = (self._owner_resolver(context), self._agent_name, task.id)
        return await self._ledger._call(lambda: self._ledger._save_task(key, task, prev_version))

    async def get(self, task_id: str, context: ServerCallContext) -> StoredTask | None:
        key = (self._owner_resolver(context), self._agent_name, task_id)

        def get_task() -> StoredTask | None:
            row = _task_row(self._ledger._db, key)
            return (
                None
                if row is None
                else StoredTask(Task.FromString(row["task_proto"]), TaskVersion(row["version"]))
            )

        return await self._ledger._call(get_task)

    async def list(self, params: ListTasksRequest, context: ServerCallContext) -> ListTasksResponse:
        filters = ["owner = ?", "agent_name = ?"]
        values: list[str | int] = [self._owner_resolver(context), self._agent_name]
        if params.context_id:
            filters.append("context_id = ?")
            values.append(params.context_id)
        if params.status:
            filters.append("state = ?")
            values.append(params.status)
        where = " AND ".join(filters)

        def list_tasks() -> ListTasksResponse:
            rows = self._ledger._db.execute(
                f"SELECT task_proto FROM a2a_tasks WHERE {where}", values
            )
            return _list_page((Task.FromString(row["task_proto"]) for row in rows), params)

        return await self._ledger._call(list_tasks)

    async def delete(self, task_id: str, context: ServerCallContext) -> None:
        key = (self._owner_resolver(context), self._agent_name, task_id)
        await self._ledger._call(lambda: self._ledger._delete_task(key))
