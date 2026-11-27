from __future__ import annotations

import asyncio
import hashlib
import io
import json
import socket
import sqlite3
import sys
import warnings
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager, closing
from pathlib import Path

import httpx
import pytest
import uvicorn
from a2a.auth.user import User
from a2a.extensions.common import HTTP_EXTENSION_HEADER
from a2a.server.context import ServerCallContext
from a2a.types import Artifact, Message, Part, Role, SendMessageRequest, Task, TaskState, TaskStatus
from a2a.utils.constants import PROTOCOL_VERSION_1_0, VERSION_HEADER
from a2a.utils.errors import TaskNotFoundError
from google.protobuf.json_format import MessageToDict
from starlette.applications import Starlette

import john.drivers
from john import (
    LIFECYCLE_EXT_URI,
    WORKSPACE_EXT_URI,
    AgentProfile,
    AuthRequiredError,
    BusyError,
    Capabilities,
    DuplicateMessageConflictError,
    John,
    OutputSpec,
    UnsupportedError,
    Workspace,
    WorkspaceGrant,
)
from john.bridge import build_agent_card
from john.cli import main
from john.coordinator import _send_fingerprint
from john.ledger import SqliteLedger
from john.types import (
    ApprovalPolicy,
    EventSink,
    StartedEvent,
    TurnFinished,
    TurnNeedsInput,
    TurnOutcome,
    TurnRequest,
    TurnResult,
    WorkspaceAccess,
)
from john.workspace import WorkspaceRegistry, verify_workspace_artifact

COMPLETED = TaskState.TASK_STATE_COMPLETED
INPUT_REQUIRED = TaskState.TASK_STATE_INPUT_REQUIRED
FAILED = TaskState.TASK_STATE_FAILED
CANCELED = TaskState.TASK_STATE_CANCELED

type Program = Callable[[FixtureSession, TurnRequest], Awaitable[TurnOutcome]]


class FixtureSession:
    native_ref = "fixture-session"
    ignored_fields: dict[str, str] = {}

    def __init__(
        self, driver: FixtureDriver, profile: AgentProfile, path: Path | None, ref: str | None
    ) -> None:
        self.driver, self.profile, self.path, self.resume_ref = driver, profile, path, ref
        self.capabilities = profile.capabilities
        self.cancelled = asyncio.Event()

    async def run_turn(
        self, request: TurnRequest, on_event: EventSink, on_approval: ApprovalPolicy
    ) -> TurnOutcome:
        self.driver.requests.append((self.profile.name, request))
        await on_event(StartedEvent(run_id=request.run_id, seq=0))
        return await self.driver.program(self, request)

    async def cancel(self, run_id: str) -> None:
        self.cancelled.set()


class FixtureDriver:
    def __init__(self, program: Program, *, auth_failures_on_open: frozenset[str] = frozenset()):
        self.program = program
        self.auth_failures_on_open = auth_failures_on_open
        self.requests: list[tuple[str, TurnRequest]] = []
        self.opened: list[tuple[str, Path | None, WorkspaceAccess, str | None]] = []

    @asynccontextmanager
    async def open(
        self,
        profile: AgentProfile,
        *,
        workspace_path: Path | None,
        access: WorkspaceAccess,
        resume_ref: str | None = None,
    ) -> AsyncIterator[FixtureSession]:
        self.opened.append((profile.name, workspace_path, access, resume_ref))
        if profile.name in self.auth_failures_on_open:
            raise AuthRequiredError("Private native login details")
        yield FixtureSession(self, profile, workspace_path, resume_ref)


def finish(request: TurnRequest, *, text: str = "") -> TurnFinished:
    return TurnFinished(result=TurnResult(run_id=request.run_id, status="completed", text=text))


async def echo(session: FixtureSession, request: TurnRequest) -> TurnOutcome:
    return finish(request, text=request.prompt)


class LocalUser(User):
    @property
    def is_authenticated(self) -> bool:
        return True

    @property
    def user_name(self) -> str:
        return "local"


def profile(name: str = "worker", **fields) -> AgentProfile:
    return AgentProfile(name=name, kind="fixture", **fields)


def write_config(directory: Path) -> Path:
    (directory / "files").mkdir(exist_ok=True)
    path = directory / "agents.toml"
    path.write_text(
        """
[ledger]
path = "state/ledger.db"

[workspaces.sample]
path = "files"
max_access = "read_write"
allowed_principals = ["local"]
allowed_agents = ["worker"]

[agents.worker]
kind = "fixture"
description = "Review sample records"
command = "unused-fixture-command"
args = ["--fixture"]

[agents.worker.capabilities]
enforced_sandbox = ["read_only", "read_write"]
task_questions = true

[[agents.worker.skills]]
id = "review"
name = "Review"
tags = ["sample"]

[agents.worker.env]
FIXTURE_MODE = "test"

[agents.worker.options]
count = 2
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return path


def make_john(
    ledger: SqliteLedger,
    driver: FixtureDriver,
    *,
    profiles: list[AgentProfile] | None = None,
    workspaces: WorkspaceRegistry | tuple[WorkspaceGrant, ...] = (),
) -> John:
    return John(
        profiles=profiles or [profile()],
        workspaces=workspaces,
        ledger=ledger,
        drivers={"fixture": driver},
    )


async def test_config_open_recovers_stale_runs_before_dispatch(tmp_path: Path) -> None:
    config = write_config(tmp_path)
    db_path = tmp_path / "state" / "ledger.db"
    db_path.parent.mkdir()
    ledger = await SqliteLedger.open(db_path)
    try:
        from a2a.server.cluster.version import TaskVersion

        await ledger.task_store_for("worker").save(
            Task(
                id="stale-task",
                context_id="old-context",
                status=TaskStatus(state=TaskState.TASK_STATE_WORKING),
            ),
            context=ServerCallContext(user=LocalUser()),
            prev_version=TaskVersion.MISSING,
        )
        await ledger.start_run(
            "stale-run",
            owner="local",
            agent_name="worker",
            task_id="stale-task",
            context_id="old-context",
        )
    finally:
        await ledger.close()

    async def inspect_profile(session: FixtureSession, request: TurnRequest) -> TurnOutcome:
        assert session.profile.profile_dir is None
        assert session.profile.command == "unused-fixture-command"
        assert session.profile.args == ("--fixture",)
        assert session.profile.env == {"FIXTURE_MODE": "test"}
        assert session.profile.options == {"count": 2}
        return finish(request, text=request.prompt)

    driver = FixtureDriver(inspect_profile)
    async with John.open(config, drivers={"fixture": driver}) as john:
        with pytest.raises(BusyError, match="Ledger is locked by another active coordinator"):
            async with John.open(config, drivers={"fixture": driver}):
                pytest.fail("A second coordinator opened the active ledger")
        card = john.list_cards()["worker"]
        assert card.skills[0].id == "review"
        assert MessageToDict(card.capabilities.extensions[0].params)["enforcedSandbox"] == [
            "read_only",
            "read_write",
        ]
        recovered = await john.get("worker", "stale-task")
        assert recovered.status.state == FAILED
        assert MessageToDict(recovered.metadata) == {LIFECYCLE_EXT_URI: {"indeterminate": True}}
        run = await john.ledger.get_run("stale-run")
        assert run is not None and run["status"] == "indeterminate"
        assert (
            await john.assign("worker", "ready", workspace=Workspace("sample"))
        ).status.state == (COMPLETED)
        assert driver.opened == [("worker", tmp_path / "files", "read_only", None)]
    with pytest.raises(RuntimeError, match="Ledger is closed"):
        await john.ledger.get_run("stale-run")


async def test_embedded_handoff_pauses_resumes_and_keeps_run_lineage(tmp_path: Path) -> None:
    source = b"record,score\na,9\n"

    async def handoff(session: FixtureSession, request: TurnRequest) -> TurnOutcome:
        if session.profile.name == "prepare":
            (output,) = request.outputs
            assert session.path == tmp_path and output.path == "prepared.csv"
            (session.path / output.path).write_bytes(source)
            return finish(request)
        if request.continuation is None:
            (reference,) = request.input_refs
            assert reference.sha256 == hashlib.sha256(source).hexdigest()
            assert session.path is not None
            assert (session.path / reference.path).read_bytes() == source
            return TurnNeedsInput(
                run_id=request.run_id,
                question="Which cutoff should I use?",
                continuation={"score": 9},
                native_session_ref="saved-qc",
            )
        assert request.continuation == {"score": 9} and request.prompt == "8"
        assert session.resume_ref == "saved-qc"
        return finish(request, text='{"flagged":1}')

    driver = FixtureDriver(handoff)
    ledger = await SqliteLedger.open(tmp_path / "ledger.db")
    workspaces = WorkspaceRegistry()
    workspaces.register(
        WorkspaceGrant(
            root_id="sample",
            host_path=tmp_path,
            max_access="read_write",
            allowed_principals=frozenset({"local"}),
            allowed_agents=frozenset({"prepare", "review"}),
        )
    )
    profiles = [
        profile("prepare", capabilities=Capabilities(enforced_sandbox=frozenset({"read_write"}))),
        profile(
            "review",
            capabilities=Capabilities(
                task_questions=True, resume=True, enforced_sandbox=frozenset({"read_only"})
            ),
        ),
    ]
    async with make_john(ledger, driver, profiles=profiles, workspaces=workspaces) as john:
        card = john.list_cards()["prepare"]
        card.name = "changed snapshot"
        assert john.list_cards()["prepare"].name == "prepare"
        prepared = await john.assign(
            "prepare",
            "Prepare the sample",
            workspace=Workspace("sample", "read_write"),
            outputs=[OutputSpec(name="prepared", path="prepared.csv", media_type="text/csv")],
        )
        assert prepared.status.state == COMPLETED
        (artifact,) = prepared.artifacts
        _, data = verify_workspace_artifact(artifact, workspaces, agent_name="review")
        assert data == source
        question = await john.assign(
            "review",
            "Review the file",
            workspace=Workspace("sample"),
            inputs=[artifact],
            parent=("prepare", prepared.id),
            schema={"type": "object", "properties": {"flagged": {"type": "integer"}}},
        )
        assert question.status.state == INPUT_REQUIRED
        assert question.status.message.parts[0].text == "Which cutoff should I use?"
        result = await john.reply("review", question, "8", message_id="cutoff")
        assert result.status.state == COMPLETED
        assert result.id == question.id and result.context_id == question.context_id
        assert MessageToDict(result.artifacts[0].parts[0].data) == {"flagged": 1}
        assert await john.reply("review", question, "8", message_id="cutoff") == result
        with pytest.raises(DuplicateMessageConflictError):
            await john.reply("review", question, "7", message_id="cutoff")
        assert await john.get("review", result.id) == result
        (preparation,) = await ledger.list_runs(task_id=prepared.id)
        paused_run, final_run = await ledger.list_runs(task_id=result.id)
        assert paused_run["parent_run_id"] == preparation["run_id"]
        assert final_run["parent_run_id"] == paused_run["run_id"]
        assert len(driver.requests) == 3


async def test_concurrent_and_persisted_retries_use_the_reserved_id(tmp_path: Path) -> None:
    path = tmp_path / "ledger.db"
    entered, release = asyncio.Event(), asyncio.Event()

    async def held(session: FixtureSession, request: TurnRequest) -> TurnOutcome:
        entered.set()
        await release.wait()
        return finish(request, text="done")

    driver = FixtureDriver(held)
    ledger = await SqliteLedger.open(path)
    article = Artifact(artifact_id="sample", parts=[Part(text="data")])
    async with make_john(ledger, driver) as john:
        initial = asyncio.create_task(
            john.assign("worker", "run", inputs=[article], message_id="m1")
        )
        try:
            async with asyncio.timeout(5):
                await entered.wait()
                retry = asyncio.create_task(
                    john.assign("worker", "run", inputs=[article], message_id="m1")
                )
                with pytest.raises(DuplicateMessageConflictError):
                    await john.assign("worker", "changed", inputs=[article], message_id="m1")
        finally:
            release.set()
        first, second = await asyncio.wait_for(asyncio.gather(initial, retry), 5)
        assert first == second
        assert [
            event.kind for event in await ledger.get_run_events(driver.requests[0][1].run_id)
        ] == ["started", "finished"]
        with closing(sqlite3.connect(path)) as sql:
            row = sql.execute(
                "SELECT task_id FROM message_reservations WHERE owner='local' AND message_id='m1'"
            ).fetchone()
        assert row == (first.id,)
        changed = Artifact(artifact_id=article.artifact_id, parts=[Part(text="different content")])
        with pytest.raises(DuplicateMessageConflictError):
            await john.assign("worker", "run", inputs=[changed], message_id="m1")
        assert len(driver.requests) == 1

    reopened = await SqliteLedger.open(path)
    async with make_john(reopened, driver) as john:
        assert await john.assign("worker", "run", inputs=[article], message_id="m1") == first
    assert len(driver.requests) == 1


async def test_open_auth_fallback_replays_one_task_across_restart(tmp_path: Path) -> None:
    path = tmp_path / "ledger.db"
    profiles = [
        profile(fallback_agents=("first", "second", "ready")),
        profile("first", fallback_agents=("second",)),
        profile("second"),
        AgentProfile(name="ready", kind="alternate"),
    ]
    primary = FixtureDriver(echo, auth_failures_on_open=frozenset({"worker", "first", "second"}))
    fallback = FixtureDriver(echo)
    drivers = {"fixture": primary, "alternate": fallback}
    ledger = await SqliteLedger.open(path)
    async with John(profiles=profiles, workspaces=(), ledger=ledger, drivers=drivers) as john:
        task = await john.assign("worker", "sample", message_id="fallback")
        assert task.status.state == COMPLETED and task.artifacts[0].parts[0].text == "sample"
        assert MessageToDict(task.metadata) == {
            LIFECYCLE_EXT_URI: {"fallback_from": "worker", "resolved_agent": "ready"}
        }
        assert await john.get("worker", task.id) == task
        with pytest.raises(TaskNotFoundError):
            await john.get("ready", task.id)
        assert await john.assign("worker", "sample", message_id="fallback") == task
        with pytest.raises(DuplicateMessageConflictError):
            await john.assign("worker", "different", message_id="fallback")
        runs = await ledger.list_runs(task_id=task.id)
        assert [run["status"] for run in runs] == ["failed", "failed", "failed", "completed"]
        assert [run["parent_run_id"] for run in runs] == [
            None,
            *[run["run_id"] for run in runs[:-1]],
        ]
        assert all(run["context_id"] == task.context_id for run in runs)
        with closing(sqlite3.connect(path)) as sql:
            assert sql.execute(
                "SELECT task_id, context_id, agent_name FROM a2a_tasks"
            ).fetchall() == [(task.id, task.context_id, "worker")]
            assert sql.execute(
                "SELECT task_id, message_id, agent_name FROM message_reservations"
            ).fetchall() == [(task.id, "fallback", "worker")]

    ledger = await SqliteLedger.open(path)
    async with John(profiles=profiles, workspaces=(), ledger=ledger, drivers=drivers) as john:
        assert await john.assign("worker", "sample", message_id="fallback") == task
    assert [name for name, *_ in primary.opened] == ["worker", "first", "second"]
    assert not primary.requests and [name for name, _ in fallback.requests] == ["ready"]


@pytest.mark.parametrize("resume_auth_required", [False, True])
async def test_reply_resumes_exact_fallback_profile_after_restart(
    tmp_path: Path,
    resume_auth_required: bool,
) -> None:
    config = tmp_path / "agents.toml"
    config.write_text(
        """
[agents.worker]
kind = "fixture"
fallback_agents = ["backup"]

[agents.backup]
kind = "alternate"
fallback_agents = ["spare"]

[agents.backup.capabilities]
task_questions = true
resume = true

[agents.spare]
kind = "fixture"
""",
        encoding="utf-8",
    )
    continuation = {"phase": "question", "__john_executor_agent__": "native data"}

    async def ask(session: FixtureSession, request: TurnRequest) -> TurnOutcome:
        assert session.profile.name == "backup"
        if request.continuation is None:
            return TurnNeedsInput(
                run_id=request.run_id,
                question="Which cutoff?",
                continuation=continuation,
                native_session_ref="backup-session",
            )
        assert session.resume_ref == "backup-session"
        assert request.continuation == continuation
        return finish(request, text=f"accepted {request.prompt}")

    primary = FixtureDriver(echo, auth_failures_on_open=frozenset({"worker"}))
    backup = FixtureDriver(ask)
    drivers = {"fixture": primary, "alternate": backup}
    resolution = {"fallback_from": "worker", "resolved_agent": "backup"}
    async with John.open(config, drivers=drivers) as john:
        parked = await john.assign("worker", "ask", message_id="initial")
        assert (
            parked.status.state == INPUT_REQUIRED
            and parked.status.message.parts[0].text == "Which cutoff?"
        )
        assert MessageToDict(parked.metadata)[LIFECYCLE_EXT_URI] == resolution
        assert await john.get("worker", parked.id) == parked
        checkpoint = await john.ledger.peek_continuation(
            "local", "worker", parked.id, parked.context_id
        )
        assert (
            checkpoint is not None
            and checkpoint["continuation"]["__john_executor_agent__"] == "backup"
        )

    if resume_auth_required:
        backup.auth_failures_on_open = frozenset({"backup"})
    async with John.open(config, drivers=drivers) as john:
        task = await john.reply("worker", parked, "5", message_id="answer")
        assert (task.id, task.context_id) == (parked.id, parked.context_id)
        assert await john.get("worker", task.id) == task
        assert await john.reply("worker", parked, "5", message_id="answer") == task
        if resume_auth_required:
            assert (
                task.status.state == FAILED and task.status.message.parts[0].text == "auth_required"
            )
            assert MessageToDict(task.metadata)[LIFECYCLE_EXT_URI] == resolution | {
                "auth_required": True
            }
        else:
            assert (
                task.status.state == COMPLETED and task.artifacts[0].parts[0].text == "accepted 5"
            )
            assert MessageToDict(task.metadata)[LIFECYCLE_EXT_URI] == resolution
            _, paused_run, resumed = await john.ledger.list_runs(task_id=task.id)
            assert resumed["parent_run_id"] == paused_run["run_id"]
    assert [name for name, *_ in primary.opened] == ["worker"]
    assert backup.opened == [
        ("backup", None, "read_only", None),
        ("backup", None, "read_only", "backup-session"),
    ]


@pytest.mark.parametrize(
    ("profiles", "error", "match"),
    [
        pytest.param(
            [profile(fallback_agents=("missing",))],
            ValueError,
            "Unknown fallback agent",
            id="unknown",
        ),
        pytest.param(
            [profile(fallback_agents=("worker",))], ValueError, "Fallback agent cycle", id="self"
        ),
        pytest.param(
            [profile(fallback_agents=("backup",)), profile("backup", fallback_agents=("worker",))],
            ValueError,
            "Fallback agent cycle",
            id="cycle",
        ),
        pytest.param(
            [
                profile(fallback_agents=("peer",)),
                AgentProfile(name="peer", kind="remote", url="http://127.0.0.1/a2a"),
            ],
            UnsupportedError,
            "Fallback agents must be local profiles",
            id="remote-target",
        ),
        pytest.param(
            [
                profile(f"agent-{number}", fallback_agents=(f"agent-{number + 1}",))
                for number in range(6)
            ]
            + [profile("agent-6")],
            ValueError,
            "at most 5 levels",
            id="depth",
        ),
    ],
)
async def test_invalid_fallback_configuration_is_rejected(
    tmp_path: Path,
    profiles: list[AgentProfile],
    error: type[Exception],
    match: str,
) -> None:
    ledger = await SqliteLedger.open(tmp_path / "ledger.db")
    driver = FixtureDriver(echo)
    try:
        with pytest.raises(error, match=match):
            make_john(ledger, driver, profiles=profiles)
        assert not driver.opened
    finally:
        await ledger.close()


async def test_retry_can_join_dispatch_after_cancelled_reservation_wait(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger = await SqliteLedger.open(tmp_path / "ledger.db")
    driver = FixtureDriver(echo)
    committed, release = asyncio.Event(), asyncio.Event()
    reserve = ledger.reserve_message

    async def held_reservation(*args, **kwargs):
        result = await reserve(*args, **kwargs)
        if result[1]:
            committed.set()
            await release.wait()
        return result

    monkeypatch.setattr(ledger, "reserve_message", held_reservation)
    async with make_john(ledger, driver) as john:
        first = asyncio.create_task(john.assign("worker", "sample", message_id="retry"))
        try:
            await asyncio.wait_for(committed.wait(), 5)
            first.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first
        finally:
            release.set()
        result = await asyncio.wait_for(john.assign("worker", "sample", message_id="retry"), 5)
        assert result.status.state == COMPLETED and len(driver.requests) == 1


@asynccontextmanager
async def rpc_client(john: John) -> AsyncIterator[httpx.AsyncClient]:
    app = john.build_http_app("worker", url="http://127.0.0.1/rpc")
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://127.0.0.1",
            headers={VERSION_HEADER: PROTOCOL_VERSION_1_0},
        ) as client,
    ):
        yield client


def wire_message(message_id: str = "retry", *, text: str = "sample") -> Message:
    return Message(message_id=message_id, role=Role.ROLE_USER, parts=[Part(text=text)])


async def send_rpc(
    client: httpx.AsyncClient,
    message: Message,
    *,
    stream: bool = False,
    workspace: bool = False,
) -> list[dict]:
    response = await client.post(
        "/rpc",
        json={
            "jsonrpc": "2.0",
            "id": "request",
            "method": "SendStreamingMessage" if stream else "SendMessage",
            "params": MessageToDict(SendMessageRequest(message=message)),
        },
        headers={HTTP_EXTENSION_HEADER: WORKSPACE_EXT_URI} if workspace else None,
    )
    assert response.status_code == 200
    if response.headers["content-type"].startswith("text/event-stream"):
        return [
            json.loads(line[5:]) for line in response.text.splitlines() if line.startswith("data:")
        ]
    return [response.json()]


@pytest.mark.parametrize("first_streams", [False, True])
async def test_http_retries_join_one_scoped_dispatch(tmp_path: Path, first_streams: bool) -> None:
    ledger = await SqliteLedger.open(tmp_path / "ledger.db")
    started, release = asyncio.Event(), asyncio.Event()

    async def hold(session: FixtureSession, request: TurnRequest) -> TurnOutcome:
        started.set()
        await release.wait()
        return await echo(session, request)

    driver = FixtureDriver(hold)
    async with rpc_client(make_john(ledger, driver)) as client:
        first = asyncio.create_task(send_rpc(client, wire_message(), stream=first_streams))
        try:
            async with asyncio.timeout(5):
                await started.wait()
                counter = asyncio.create_task(
                    send_rpc(client, wire_message(), stream=not first_streams)
                )
                (conflict,) = await send_rpc(client, wire_message(text="different"))
                assert conflict["error"]["code"] == -32602
                release.set()
                initial, duplicate = await asyncio.gather(first, counter)
        finally:
            release.set()
            await asyncio.gather(first, return_exceptions=True)
        (run,) = await ledger.list_runs()
        assert len(driver.requests) == 1 and run["status"] == "completed"
        assert initial[0]["result"]["task"]["id"] == run["task_id"]
        (retry,) = duplicate
        assert retry["result"]["task"]["id"] == run["task_id"]
        assert retry["result"]["task"]["status"]["state"] == "TASK_STATE_COMPLETED"
        if first_streams:
            assert initial[-2]["result"]["artifactUpdate"]["artifact"]["parts"] == [
                {"text": "sample", "mediaType": "text/plain"}
            ]
            assert (
                initial[-1]["result"]["statusUpdate"]["status"]["state"] == "TASK_STATE_COMPLETED"
            )
        for stream in (False, True):
            (replay,) = await send_rpc(client, wire_message(), stream=stream)
            assert replay["result"]["task"] == retry["result"]["task"]
        assert await ledger.list_runs() == [run]


async def test_restart_retries_only_provably_unsent_orphan_reservations(tmp_path: Path) -> None:
    path = tmp_path / "ledger.db"
    ledger = await SqliteLedger.open(path)
    orphan, dispatched = wire_message("orphan"), wire_message("dispatched")
    call = ServerCallContext()
    for message in (orphan, dispatched):
        fingerprint = _send_fingerprint(SendMessageRequest(message=message), call)
        await ledger.reserve_message(
            "", "worker", message.message_id, fingerprint, message.message_id
        )
    await ledger.start_run(
        "prior-run", owner="", agent_name="worker", task_id="dispatched", context_id="context"
    )
    await ledger.close()

    reopened = await SqliteLedger.open(path)
    driver = FixtureDriver(echo)
    async with rpc_client(make_john(reopened, driver)) as client:
        (uncertain,) = await send_rpc(client, dispatched)
        assert uncertain["error"]["code"] == -32001
        (result,) = await send_rpc(client, orphan)
        assert result["result"]["task"]["id"] == "orphan"
        assert len(driver.requests) == 1
        prior = await reopened.get_run("prior-run")
        assert prior is not None and prior["status"] == "indeterminate"


async def test_http_reply_reauthorizes_workspace_before_consuming_checkpoint(
    tmp_path: Path,
) -> None:
    path = tmp_path / "ledger.db"
    caps = Capabilities(task_questions=True, enforced_sandbox=frozenset({"read_only"}))
    profiles = [profile(require_workspace=True, capabilities=caps)]

    async def ask(session: FixtureSession, request: TurnRequest) -> TurnOutcome:
        if request.continuation is None:
            return TurnNeedsInput(
                run_id=request.run_id, question="Which cutoff?", continuation={"sample": 1}
            )
        return await echo(session, request)

    driver = FixtureDriver(ask)
    registry = WorkspaceRegistry()
    grant = WorkspaceGrant(root_id="sample", host_path=tmp_path)
    registry.register(grant)
    ledger = await SqliteLedger.open(path)
    async with rpc_client(
        make_john(ledger, driver, profiles=profiles, workspaces=registry)
    ) as client:
        message = wire_message()
        message.metadata.update({WORKSPACE_EXT_URI: {"root": "sample"}})
        (response,) = await send_rpc(client, message, workspace=True)
        parked = response["result"]["task"]
        reply = wire_message("answer", text="5")
        reply.task_id, reply.context_id = parked["id"], parked["contextId"]
        (extension_error,) = await send_rpc(client, reply)
        assert "required workspace extension" in extension_error["error"]["message"]
        registry.register(grant.model_copy(update={"allowed_principals": frozenset({"local"})}))
        (grant_error,) = await send_rpc(client, reply, workspace=True)
        assert grant_error["error"]["code"] == -32602
        assert "unauthorized" in grant_error["error"]["message"]
        checkpoint = await ledger.peek_continuation("", "worker", reply.task_id, reply.context_id)
        assert checkpoint is not None and checkpoint["continuation"] == {"sample": 1}
        assert len(driver.requests) == 1
        registry.register(grant)
        (result,) = await send_rpc(client, reply, workspace=True)
        assert result["result"]["task"]["status"]["state"] == "TASK_STATE_COMPLETED"
        assert len(driver.requests) == 2


@asynccontextmanager
async def loopback(john: John, agent: str) -> AsyncIterator[str]:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        url = f"http://127.0.0.1:{listener.getsockname()[1]}/rpc"
        app = john.build_http_app(agent, url=url)
        server = uvicorn.Server(
            uvicorn.Config(app, log_level="error", access_log=False, timeout_graceful_shutdown=3)
        )
        serving = asyncio.create_task(server.serve(sockets=[listener]))
        try:
            async with asyncio.timeout(5):
                while not server.started:
                    if serving.done():
                        serving.result()
                        raise RuntimeError("Loopback server exited before startup")
                    await asyncio.sleep(0.01)
            yield url
        finally:
            server.should_exit = True
            await asyncio.wait_for(serving, 5)


@pytest.mark.parametrize("first_streams", [False, True])
async def test_real_loopback_fallback_reuses_the_public_task_for_get_and_retry(
    tmp_path: Path,
    first_streams: bool,
) -> None:
    ledger = await SqliteLedger.open(tmp_path / "ledger.db")
    driver = FixtureDriver(echo, auth_failures_on_open=frozenset({"worker"}))
    profiles = [profile(fallback_agents=("backup",)), profile("backup")]
    async with loopback(make_john(ledger, driver, profiles=profiles), "worker") as url:
        async with httpx.AsyncClient(
            base_url=httpx.URL(url).copy_with(path="/"),
            headers={VERSION_HEADER: PROTOCOL_VERSION_1_0},
            timeout=15,
        ) as client:
            initial = await send_rpc(client, wire_message("fallback"), stream=first_streams)
            task_id = initial[0]["result"]["task"]["id"]
            public_context = initial[0]["result"]["task"]["contextId"]
            response = await client.post(
                "/rpc",
                json={
                    "jsonrpc": "2.0",
                    "id": "get",
                    "method": "GetTask",
                    "params": {"id": task_id},
                },
            )
            assert response.status_code == 200
            task = response.json()["result"]
            assert task["id"] == task_id and task["contextId"] == public_context
            assert task["status"]["state"] == "TASK_STATE_COMPLETED"
            resolution = {"fallback_from": "worker", "resolved_agent": "backup"}
            assert task["metadata"][LIFECYCLE_EXT_URI] == resolution
            if first_streams:
                statuses = [
                    item["result"]["statusUpdate"]
                    for item in initial
                    if "statusUpdate" in item["result"]
                ]
                assert [status["status"]["state"] for status in statuses] == [
                    "TASK_STATE_WORKING",
                    "TASK_STATE_COMPLETED",
                ]
                assert all(status["taskId"] == task_id for status in statuses)
                assert statuses[-1]["metadata"][LIFECYCLE_EXT_URI] == resolution
            (replay,) = await send_rpc(client, wire_message("fallback"), stream=not first_streams)
            assert replay["result"]["task"] == task
        assert [run["status"] for run in await ledger.list_runs(task_id=task_id)] == [
            "failed",
            "completed",
        ]
    assert [name for name, *_ in driver.opened] == ["worker", "backup"]
    assert [name for name, _ in driver.requests] == ["backup"]


async def test_remote_a2a_assign_reply_get_and_cancel_over_real_loopback(tmp_path: Path) -> None:
    async def remote_program(session: FixtureSession, request: TurnRequest) -> TurnOutcome:
        assert session.path == tmp_path
        if request.continuation is None:
            return TurnNeedsInput(
                run_id=request.run_id, question="Pick a number", continuation={"answer": True}
            )
        return finish(request, text=f"accepted {request.prompt}")

    server_path, client_path = tmp_path / "server.db", tmp_path / "client.db"
    server_ledger = await SqliteLedger.open(server_path)
    driver = FixtureDriver(remote_program)
    server_john = make_john(
        server_ledger,
        driver,
        profiles=[
            profile(
                require_workspace=True,
                capabilities=Capabilities(
                    task_questions=True, enforced_sandbox=frozenset({"read_only"})
                ),
            )
        ],
        workspaces=(WorkspaceGrant(root_id="sample", host_path=tmp_path),),
    )
    async with loopback(server_john, "worker") as url:
        remote_profile = AgentProfile(name="peer", kind="remote", url=url)
        remote_ledger = await SqliteLedger.open(client_path)
        async with John(
            profiles={"peer": remote_profile}, workspaces=(), ledger=remote_ledger, drivers={}
        ) as client:
            card = client.list_cards()["peer"]
            assert card.supported_interfaces[0].url == url
            assert card.capabilities.extensions[0].uri == WORKSPACE_EXT_URI
            assert card.capabilities.extensions[0].required
            with pytest.raises(UnsupportedError, match="requires a workspace"):
                await client.assign("peer", "ask")
            with pytest.raises(UnsupportedError, match="cannot enforce"):
                await client.assign("peer", "ask", workspace=Workspace("sample", "read_write"))
            assert driver.requests == []
            async with asyncio.timeout(15):
                paused = await client.assign("peer", "ask", workspace=Workspace("sample"))
                assert paused.status.state == INPUT_REQUIRED
                assert await client.get("peer", paused.id) == paused
                with pytest.raises(TaskNotFoundError):
                    await server_john.get("worker", paused.id)
                result = await client.reply("peer", paused, "17")
                assert result.status.state == COMPLETED and result.id == paused.id
                assert result.artifacts[0].parts[0].text == "accepted 17"
                assert await client.get("peer", result.id) == result

                another = await client.assign("peer", "park", workspace=Workspace("sample"))
                cancelled = await client.cancel("peer", another.id)
                assert cancelled.status.state == CANCELED
                assert await client.get("peer", another.id) == cancelled
            assert await remote_ledger.list_runs() == []
        with pytest.raises(RuntimeError, match="Ledger is closed"):
            await remote_ledger.list_runs()
    with pytest.raises(RuntimeError, match="Ledger is closed"):
        await server_ledger.list_runs()


async def test_remote_message_only_response_returns_completed_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    iface = "https://peer.invalid/rpc"
    card = build_agent_card(profile(), url=iface)
    seen = []

    async def handle(request: httpx.Request) -> httpx.Response:
        assert request.headers["A2A-Version"] == "1.0"
        seen.append(request.url.path)
        if request.method == "GET":
            return httpx.Response(200, json=MessageToDict(card))
        body = json.loads(request.content)
        assert body["method"] == "SendMessage"
        message_id = body["params"]["message"]["messageId"]
        return httpx.Response(
            200,
            json={
                "jsonrpc": "2.0",
                "id": body["id"],
                "result": {
                    "message": {
                        "messageId": f"answer-{message_id}",
                        "contextId": "context",
                        "role": "ROLE_AGENT",
                        "parts": [{"text": "direct response"}],
                    }
                },
            },
        )

    async_client = httpx.AsyncClient
    opened = []

    def fake_client(**options) -> httpx.AsyncClient:
        client = async_client(transport=httpx.MockTransport(handle), **options)
        opened.append(client)
        return client

    monkeypatch.setattr(httpx, "AsyncClient", fake_client)
    monkeypatch.setattr("john.coordinator._MAX_DIRECT_RESPONSES", 1)
    ledger = await SqliteLedger.open(tmp_path / "ledger.db")
    async with John(
        profiles=[AgentProfile(name="peer", kind="remote", url=iface)],
        workspaces=(),
        ledger=ledger,
        drivers={},
    ) as john:
        task = await john.assign("peer", "say hello", message_id="message-only")
        assert (task.context_id, task.status.state) == ("context", COMPLETED)
        assert task.status.message.message_id == "answer-message-only"
        assert task.status.message.parts[0].text == "direct response"
        assert await john.get("peer", task.id) == task
        task.status.message.parts[0].text = "client mutation"
        saved = await john.get("peer", task.id)
        assert saved.status.message.parts[0].text == "direct response"
        with pytest.raises(
            UnsupportedError, match="Remote peer returned a direct message with no task"
        ):
            await john.cancel("peer", task.id)
        other = await john.assign("peer", "say hello again", message_id="other")
        assert await john.get("peer", other.id) == other
        with pytest.raises(TaskNotFoundError, match="snapshot is no longer available"):
            await john.get("peer", task.id)
        with pytest.raises(
            UnsupportedError, match="Remote peer returned a direct message with no task"
        ):
            await john.cancel("peer", task.id)
    assert seen == ["/.well-known/agent-card.json", "/rpc", "/rpc"]
    assert opened[0].is_closed


def cli_args(tmp_path: Path, command: str = "run") -> list[str]:
    return [command, "--config", str(write_config(tmp_path)), "--agent", "worker"]


def install_fixture_driver(monkeypatch: pytest.MonkeyPatch, program: Program = echo) -> None:
    driver = FixtureDriver(program)
    monkeypatch.setattr(john.drivers, "default_drivers", lambda: {"fixture": driver}, raising=False)


def test_cli_run_reads_stdin_and_separates_text_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    install_fixture_driver(monkeypatch)
    monkeypatch.setattr(sys, "stdin", io.StringIO("sample from pipe"))
    assert main(cli_args(tmp_path)) == 0
    captured = capsys.readouterr()
    assert captured.out == "sample from pipe\n"
    assert "Started run " in captured.err


def test_cli_run_emits_ndjson_events_and_final_task(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def write(session: FixtureSession, request: TurnRequest) -> TurnOutcome:
        assert session.path is not None
        (session.path / request.outputs[0].path).write_bytes(b"written")
        return finish(request)

    install_fixture_driver(monkeypatch, write)
    caplog.set_level("WARNING", logger="a2a.server.request_handlers.default_request_handler_v2")
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        assert (
            main(
                [
                    *cli_args(tmp_path),
                    "--json",
                    "--workspace",
                    "sample:read_write",
                    "--output",
                    "result:result.txt:text/plain",
                    "write the file",
                ]
            )
            == 0
        )
    assert not any("cross-replica streaming is disabled" in str(item.message) for item in recorded)
    assert "cross-replica streaming is disabled" not in caplog.text
    captured = capsys.readouterr()
    assert captured.err == ""
    records = [json.loads(line) for line in captured.out.splitlines()]
    assert any(event.get("kind") == "started" for event in records[:-1])
    task = records[-1]
    assert task["status"]["state"] == "TASK_STATE_COMPLETED"
    assert task["artifacts"][0]["parts"][0]["data"]["path"] == "result.txt"


@pytest.mark.parametrize("rejected", [False, True])
def test_cli_run_returns_nonzero_for_failed_or_rejected_tasks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    rejected: bool,
) -> None:
    async def fail(session: FixtureSession, request: TurnRequest) -> TurnOutcome:
        return TurnFinished(
            result=TurnResult(run_id=request.run_id, status="failed", reason="fixture failure")
        )

    install_fixture_driver(monkeypatch, fail)
    workspace = ["--workspace", "unknown:read_only"] if rejected else []
    assert main([*cli_args(tmp_path), *workspace, "run"]) == 1
    result = capsys.readouterr()
    assert result.out == ""
    assert ("TASK_STATE_REJECTED" if rejected else "TASK_STATE_FAILED") in result.err


def test_cli_serve_builds_app_with_loopback_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_fixture_driver(monkeypatch)
    seen = []

    class Server:
        def __init__(self, config: uvicorn.Config) -> None:
            self.config = config
            self.started = False

        async def serve(self) -> None:
            config = self.config
            assert config.host == "127.0.0.1" and config.port == 8080
            app = config.app
            assert isinstance(app, Starlette)
            async with app.router.lifespan_context(app):
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:8080"
                ) as client:
                    response = await client.get("/.well-known/agent-card.json")
                    assert response.status_code == 200
                    card = response.json()
                    seen.append(card["supportedInterfaces"][0]["url"])
                self.started = True

    monkeypatch.setattr(uvicorn, "Server", Server)
    assert main(cli_args(tmp_path, "serve")) == 0
    assert seen == ["http://127.0.0.1:8080/a2a"]


def test_cli_serve_refuses_non_loopback_before_opening_or_binding(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fail(*args, **kwargs):
        pytest.fail("Non-loopback host must be rejected before opening the coordinator or binding")

    monkeypatch.setattr(John, "open", fail)
    monkeypatch.setattr(uvicorn, "Server", fail)
    assert main(["serve", "--agent", "worker", "--host", "0.0.0.0"]) == 1
    assert "requires a loopback host" in capsys.readouterr().err


async def test_build_http_app_refuses_non_loopback_advertisements_and_clients(
    tmp_path: Path,
) -> None:
    ledger = await SqliteLedger.open(tmp_path / "ledger.db")
    async with make_john(ledger, FixtureDriver(echo)) as john:
        for url in ("http://0.0.0.0/rpc", "https://peer.invalid/rpc"):
            with pytest.raises(UnsupportedError, match="requires a loopback host"):
                john.build_http_app("worker", url=url)
        for url in (None, "http://localhost/rpc", "http://[::1]/rpc"):
            assert isinstance(john.build_http_app("worker", url=url), Starlette)
        app = john.build_http_app("worker")
        async with httpx.AsyncClient(
            base_url="http://127.0.0.1",
            transport=httpx.ASGITransport(app=app, client=("192.0.2.1", 9000)),
        ) as client:
            response = await client.get("/.well-known/agent-card.json")
        assert response.status_code == 403


def test_missing_default_registry_has_an_actionable_cli_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delattr(john.drivers, "default_drivers", raising=False)
    assert main([*cli_args(tmp_path), "test"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "No driver registered for agent 'worker'" in captured.err
