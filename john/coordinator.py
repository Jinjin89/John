"""Coordinate embedded agents and remote A2A peers through one task facade."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import tomllib
import warnings
from collections import OrderedDict
from collections.abc import AsyncGenerator, AsyncIterator, Iterable, Mapping, Sequence
from contextlib import AsyncExitStack, aclosing, asynccontextmanager
from dataclasses import dataclass
from ipaddress import ip_address
from pathlib import Path
from typing import Any, Self
from uuid import uuid4

import httpx
from a2a.auth.user import User
from a2a.client import A2ACardResolver, ClientCallContext
from a2a.client.transports.jsonrpc import JsonRpcTransport
from a2a.extensions.common import HTTP_EXTENSION_HEADER
from a2a.server.agent_execution import RequestContext, SimpleRequestContextBuilder
from a2a.server.context import ServerCallContext
from a2a.server.events import Event
from a2a.server.owner_resolver import resolve_user_scope
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.request_handlers.request_handler import validate_request_params
from a2a.server.routes import (
    DefaultServerCallContextBuilder,
    create_agent_card_routes,
    create_jsonrpc_routes,
)
from a2a.types import (
    AgentCard,
    AgentInterface,
    Artifact,
    CancelTaskRequest,
    GetTaskRequest,
    Message,
    Part,
    Role,
    SendMessageRequest,
    SubscribeToTaskRequest,
    Task,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
)
from a2a.utils.constants import AGENT_CARD_WELL_KNOWN_PATH, PROTOCOL_VERSION_1_0, VERSION_HEADER
from a2a.utils.errors import InvalidParamsError, TaskNotFoundError, UnsupportedOperationError
from a2a.utils.task import apply_history_length, validate_history_length
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.responses import PlainTextResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from john.bridge import JohnAgentExecutor, build_agent_card
from john.ledger import SqliteLedger
from john.runtime import SessionRuntime
from john.types import (
    WORKSPACE_EXT_URI,
    AgentProfile,
    ApprovalPolicy,
    BusyError,
    Driver,
    DuplicateMessageConflictError,
    EventSink,
    OutputSpec,
    TransportError,
    UnsupportedError,
    WorkspaceAccess,
    WorkspaceGrant,
    WorkspaceRequest,
    WorkspaceViolationError,
)
from john.workspace import WorkspaceRegistry

_LOCAL_OWNER = "local"
_RESERVED_TASK_ID = "john_reserved_task_id"
_DEFAULT_SERVE_URL = "http://127.0.0.1:8080/a2a"
_MAX_STREAM_EVENTS = 256
_MAX_DIRECT_RESPONSES = 256
_DIRECT_RESPONSE_PREFIX = "john:direct-message:"
_MAX_FALLBACK_DEPTH = 5

type Parent = tuple[str, str] | str


@dataclass(frozen=True, slots=True)
class Workspace:
    """Select a registered workspace and an access level for an assignment."""

    root: str
    access: WorkspaceAccess = "read_only"


class _LocalUser(User):
    @property
    def is_authenticated(self) -> bool:
        return True

    @property
    def user_name(self) -> str:
        return _LOCAL_OWNER


class _ReservedTaskContextBuilder(SimpleRequestContextBuilder):
    async def build(
        self,
        context: ServerCallContext,
        params: SendMessageRequest | None = None,
        task_id: str | None = None,
        context_id: str | None = None,
        task: Task | None = None,
    ) -> RequestContext:
        # The SDK interprets a task ID on the incoming message as a lookup.
        # Supply newly reserved IDs after that check, during context creation.
        return await super().build(
            context, params, task_id or context.state.get(_RESERVED_TASK_ID), context_id, task
        )


@dataclass(frozen=True, slots=True)
class _RemoteAgent:
    transport: JsonRpcTransport
    tenant: str


def _is_loopback(host: str) -> bool:
    try:
        return ip_address(host).is_loopback
    except ValueError:
        return host.lower() == "localhost"


def require_loopback_host(host: str) -> None:
    if not _is_loopback(host):
        raise UnsupportedError("Unauthenticated A2A serving requires a loopback host")


class _LoopbackOnly:
    def __init__(self, app: ASGIApp) -> None:
        self._app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        peer = scope.get("client")
        if scope["type"] == "http" and (peer is None or not _is_loopback(peer[0])):
            response = PlainTextResponse("A2A access requires a loopback client", status_code=403)
            await response(scope, receive, send)
            return
        await self._app(scope, receive, send)


def _url(value: str) -> httpx.URL:
    url = httpx.URL(value)
    if url.scheme not in {"http", "https"} or not url.host or url.username or url.password:
        raise UnsupportedError("An absolute HTTP(S) URL without user information is required")
    return url


def _default_drivers() -> Mapping[str, Driver]:
    from john import drivers

    factory = getattr(drivers, "default_drivers", None)
    return {} if factory is None else factory()


def _read_config(config_path: Path) -> tuple[list[AgentProfile], WorkspaceRegistry, Path]:
    with config_path.open("rb") as file:
        config = tomllib.load(file)
    base = config_path.parent
    workspaces = WorkspaceRegistry()
    try:
        for name, settings in config.get("workspaces", {}).items():
            fields = dict(settings)
            path = base / Path(fields.pop("path")).expanduser()
            workspaces.register(WorkspaceGrant(**(fields | {"root_id": name, "host_path": path})))
        profiles = []
        for name, settings in config.get("agents", {}).items():
            fields = dict(settings)
            raw_dir = fields.pop("profile_dir", None)
            directory = None if raw_dir is None else base / Path(raw_dir).expanduser()
            profiles.append(AgentProfile(**(fields | {"name": name, "profile_dir": directory})))
        db_path = base / Path(config.get("ledger", {}).get("path", ".john/ledger.db")).expanduser()
    except (AttributeError, KeyError, TypeError, ValueError):
        raise ValueError("Invalid ledger, workspace, or agent configuration") from None
    return profiles, workspaces, db_path


def _fallback_orders(profiles: Mapping[str, AgentProfile]) -> dict[str, tuple[str, ...]]:
    resolved: dict[str, tuple[tuple[str, ...], int]] = {}
    visiting: set[str] = set()

    def visit(profile: AgentProfile) -> tuple[tuple[str, ...], int]:
        name = profile.name
        if name in visiting:
            raise ValueError(f"Fallback agent cycle involving {name!r}")
        if name in resolved:
            return resolved[name]
        if len(visiting) > _MAX_FALLBACK_DEPTH:
            raise ValueError(f"Fallback chains may have at most {_MAX_FALLBACK_DEPTH} levels")
        if profile.kind == "remote" and profile.fallback_agents:
            raise UnsupportedError("Only local profiles can configure fallback agents")

        visiting.add(name)
        ordered: dict[str, None] = {}
        depth = 0
        for target in profile.fallback_agents:
            if target not in profiles:
                raise ValueError(f"Unknown fallback agent {target!r} for {name!r}")
            if profiles[target].kind == "remote":
                raise UnsupportedError("Fallback agents must be local profiles")
            descendants, distance = visit(profiles[target])
            depth = max(depth, distance + 1)
            if depth > _MAX_FALLBACK_DEPTH:
                raise ValueError(f"Fallback chains may have at most {_MAX_FALLBACK_DEPTH} levels")
            ordered[target] = None
            ordered.update(dict.fromkeys(descendants))
        visiting.remove(name)
        result = resolved[name] = (tuple(ordered), depth)
        return result

    return {name: visit(profile)[0] for name, profile in profiles.items()}


def _fingerprint(
    message: Message,
    *,
    input_ids: Sequence[str] = (),
    schema: dict[str, Any] | None = None,
    parent: Parent | None = None,
    timeout_s: float | None = None,
) -> str:
    content = Message()
    content.CopyFrom(message)
    content.ClearField("message_id")
    options = json.dumps(
        {
            "input_ids": input_ids,
            "schema": schema,
            "parent": parent,
            "timeout_s": timeout_s,
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    digest = hashlib.sha256()
    digest.update(options.encode())
    digest.update(b"\0")
    digest.update(content.SerializeToString(deterministic=True))
    return digest.hexdigest()


def _call_context(
    extensions: Iterable[str] = (),
    *,
    schema: dict[str, Any] | None = None,
    parent_run_id: str | None = None,
    parent: Parent | None = None,
    input_ids: Sequence[str] = (),
    policy: ApprovalPolicy | None = None,
    on_event: EventSink | None = None,
    timeout_s: float | None = None,
) -> ServerCallContext:
    state = {
        "john_schema": schema,
        "john_parent_run_id": parent_run_id,
        "john_parent": parent,
        "john_input_ids": input_ids,
        "john_policy": policy,
        "john_on_event": on_event,
        "john_timeout_s": timeout_s,
    }
    return ServerCallContext(
        user=_LocalUser(),
        requested_extensions=set(extensions),
        state={key: value for key, value in state.items() if value is not None},
    )


def _peer_context(*, workspace: bool = False, timeout_s: float | None = None) -> ClientCallContext:
    return ClientCallContext(
        timeout=timeout_s,
        service_parameters={HTTP_EXTENSION_HEADER: WORKSPACE_EXT_URI} if workspace else None,
    )


def _workspace_history(task: Task) -> bool:
    return WORKSPACE_EXT_URI in task.metadata.fields or any(
        WORKSPACE_EXT_URI in item.extensions or WORKSPACE_EXT_URI in item.metadata.fields
        for item in task.history
    )


def _observe_exception(task: asyncio.Future[Any]) -> None:
    if not task.cancelled():
        task.exception()


def _send_fingerprint(params: SendMessageRequest, call: ServerCallContext) -> str:
    if isinstance(call.user, _LocalUser):
        return _fingerprint(
            params.message,
            input_ids=call.state["john_input_ids"],
            schema=call.state.get("john_schema"),
            parent=call.state.get("john_parent"),
            timeout_s=call.state.get("john_timeout_s"),
        )
    request = SendMessageRequest()
    request.CopyFrom(params)
    request.message.ClearField("message_id")
    request.configuration.ClearField("history_length")
    request.configuration.ClearField("return_immediately")
    if not request.configuration.ListFields():
        request.ClearField("configuration")
    digest = hashlib.sha256(request.SerializeToString(deterministic=True))
    digest.update(b"\0")
    digest.update(json.dumps(sorted(call.requested_extensions)).encode())
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class _Admission:
    task_id: str
    job: asyncio.Task[Task] | None = None
    events: asyncio.Queue[Event | None] | None = None


_A2A_HANDLER_LOGGER = "a2a.server.request_handlers.default_request_handler_v2"
_CROSS_REPLICA_WARNING = "cross-replica streaming is disabled"


class _SuppressCrossReplicaWarning(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return _CROSS_REPLICA_WARNING not in record.getMessage()


class _ReservedRequestHandler(DefaultRequestHandler):
    """Admit both forms of SendMessage before delegating to the SDK handler."""

    def __init__(self, ledger: SqliteLedger, executor: JohnAgentExecutor, card: AgentCard) -> None:
        self._ledger = ledger
        self._executor = executor
        self._name = card.name
        self._streaming = card.capabilities.streaming
        self._store = ledger.task_store_for(self._name)
        # John owns an exclusive single-process SQLite ledger; cross-replica streaming does not
        # apply.
        handler_logger = logging.getLogger(_A2A_HANDLER_LOGGER)
        warning_filter = _SuppressCrossReplicaWarning()
        handler_logger.addFilter(warning_filter)
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message=f".*{_CROSS_REPLICA_WARNING}.*", category=UserWarning
                )
                super().__init__(
                    agent_executor=executor,
                    task_store=self._store,
                    agent_card=card,
                    request_context_builder=_ReservedTaskContextBuilder(),
                )
        finally:
            handler_logger.removeFilter(warning_filter)
        self._admission_lock = asyncio.Lock()
        self._reservations: set[asyncio.Task[_Admission]] = set()
        self._pending: dict[tuple[str, str], asyncio.Task[Task]] = {}
        self._closing = False

    async def aclose(self) -> None:
        self._closing = True
        await asyncio.gather(*self._reservations, return_exceptions=True)
        try:
            await super().aclose()
        finally:
            pending = list(self._pending.values())
            for job in pending:
                job.cancel()
            await asyncio.gather(*pending, return_exceptions=True)

    async def _admit(
        self, params: SendMessageRequest, context: ServerCallContext, *, stream: bool = False
    ) -> _Admission:
        validate_history_length(params.configuration)
        # Caller cancellation cannot separate a durable reservation from scheduling.
        reservation = asyncio.create_task(self._reserve(params, context, stream=stream))
        self._reservations.add(reservation)
        reservation.add_done_callback(self._reservations.discard)
        reservation.add_done_callback(_observe_exception)
        try:
            return await asyncio.shield(reservation)
        except DuplicateMessageConflictError as exc:
            if isinstance(context.user, _LocalUser):
                raise
            raise InvalidParamsError(str(exc)) from None

    async def _reserve(
        self, params: SendMessageRequest, context: ServerCallContext, *, stream: bool
    ) -> _Admission:
        owner = resolve_user_scope(context)
        message = params.message
        key = (owner, message.message_id)
        fingerprint = _send_fingerprint(params, context)
        async with self._admission_lock:
            if self._closing:
                raise RuntimeError("John is closed")
            task_id = await self._ledger.lookup_message_reservation(
                owner, self._name, message.message_id, fingerprint
            )
            if task_id is None:
                if message.task_id:
                    existing = await self._snapshot(message.task_id, context)
                    if message.context_id and message.context_id != existing.context_id:
                        raise InvalidParamsError("Message context ID does not match the task")
                    await self._executor.validate_reply(existing, message, context)
                task_id, created = await self._ledger.reserve_message(
                    owner,
                    self._name,
                    message.message_id,
                    fingerprint,
                    message.task_id or str(uuid4()),
                )
            else:
                created = False
            if not created:
                if (job := self._pending.get(key)) is not None:
                    return _Admission(task_id, job)
                if not await self._ledger.is_unsent_task(owner, self._name, task_id):
                    return _Admission(task_id)

            request = SendMessageRequest()
            request.CopyFrom(params)
            call = context.model_copy(
                update={"state": dict(context.state) | {_RESERVED_TASK_ID: task_id}}
            )
            events = asyncio.Queue[Event | None](_MAX_STREAM_EVENTS + 1) if stream else None
            job = self._pending[key] = asyncio.create_task(
                self._dispatch(request, call, task_id, events)
            )

            def done(future: asyncio.Task[Task]) -> None:
                self._pending.pop(key, None)
                _observe_exception(future)

            job.add_done_callback(done)
            return _Admission(task_id, job, events)

    async def _snapshot(self, task_id: str, context: ServerCallContext) -> Task:
        stored = await self._store.get(task_id, context)
        if stored is None:
            raise TaskNotFoundError()
        return stored.task

    async def _dispatch(
        self,
        request: SendMessageRequest,
        context: ServerCallContext,
        task_id: str,
        events: asyncio.Queue[Event | None] | None,
    ) -> Task:
        if events is None:
            request.configuration.ClearField("history_length")
            result = await super().on_message_send(request, context)
            if not isinstance(result, Task) or result.id != task_id:
                raise TransportError("Embedded handler did not return the reserved task")
            return result
        try:
            congested = False
            async with aclosing(
                self._with_committed_artifacts(
                    super().on_message_send_stream(request, context), context
                )
            ) as stream:
                async for event in stream:
                    # Reserve room for the sentinel and finish execution if the client falls behind.
                    if events.qsize() == _MAX_STREAM_EVENTS:
                        congested = True
                    if not congested:
                        events.put_nowait(event)
            return await self._snapshot(task_id, context)
        finally:
            events.put_nowait(None)

    async def _result(self, admission: _Admission, context: ServerCallContext) -> Task:
        if admission.job is not None:
            return await asyncio.shield(admission.job)
        return await self._snapshot(admission.task_id, context)

    @validate_request_params
    async def on_message_send(self, params: SendMessageRequest, context: ServerCallContext) -> Task:
        admission = await self._admit(params, context)
        return apply_history_length(await self._result(admission, context), params.configuration)

    @validate_request_params
    async def on_message_send_stream(
        self, params: SendMessageRequest, context: ServerCallContext
    ) -> AsyncGenerator[Event, None]:
        if not self._streaming:
            raise UnsupportedOperationError("Streaming is not supported by the agent")
        admission = await self._admit(params, context, stream=True)
        last = None
        if admission.events is not None:
            while (event := await admission.events.get()) is not None:
                last = event
                yield event
        result = await self._result(admission, context)
        if not isinstance(last, TaskStatusUpdateEvent) or last.status.state != result.status.state:
            yield apply_history_length(result, params.configuration)

    async def on_subscribe_to_task(
        self, params: SubscribeToTaskRequest, context: ServerCallContext
    ) -> AsyncGenerator[Event, None]:
        async with aclosing(
            self._with_committed_artifacts(super().on_subscribe_to_task(params, context), context)
        ) as stream:
            async for event in stream:
                yield event

    async def _with_committed_artifacts(
        self, events: AsyncGenerator[Event, None], context: ServerCallContext
    ) -> AsyncGenerator[Event, None]:
        async with aclosing(events):
            async for event in events:
                if isinstance(event, TaskStatusUpdateEvent) and (
                    event.status.state == TaskState.TASK_STATE_COMPLETED
                ):
                    task = await self._snapshot(event.task_id, context)
                    for artifact in task.artifacts:
                        yield TaskArtifactUpdateEvent(
                            task_id=task.id,
                            context_id=task.context_id,
                            artifact=artifact,
                            append=False,
                            last_chunk=True,
                        )
                yield event


class John:
    """Own a set of local handlers, remote clients, and their ledger lifetime."""

    def __init__(
        self,
        *,
        profiles: Mapping[str, AgentProfile] | Iterable[AgentProfile],
        workspaces: WorkspaceRegistry | Iterable[WorkspaceGrant],
        ledger: SqliteLedger,
        drivers: Mapping[str, Driver] | None = None,
    ) -> None:
        if not ledger.exclusive:
            raise BusyError("John requires exclusive ledger ownership")
        self.ledger = ledger
        self._profiles: dict[str, AgentProfile] = {}
        entries = profiles.values() if isinstance(profiles, Mapping) else profiles
        for profile in entries:
            if profile.name in self._profiles:
                raise ValueError(f"Duplicate agent profile {profile.name!r}")
            self._profiles[profile.name] = profile
        fallback_orders = _fallback_orders(self._profiles)
        if isinstance(workspaces, WorkspaceRegistry):
            self._workspaces = workspaces
        else:
            self._workspaces = WorkspaceRegistry()
            for grant in workspaces:
                self._workspaces.register(grant)

        drivers = _default_drivers() if drivers is None else drivers
        self._runtime = SessionRuntime(ledger)
        self._handlers: dict[str, _ReservedRequestHandler] = {}
        self._remotes: dict[str, _RemoteAgent] = {}
        self._direct_responses: OrderedDict[tuple[str, str], Task] = OrderedDict()
        self._cards: dict[str, AgentCard] = {}
        self._http: httpx.AsyncClient | None = None
        self._lifecycle_lock = asyncio.Lock()
        self._started = self._closed = False

        local_drivers: dict[str, Driver] = {}
        for profile in self._profiles.values():
            if profile.kind == "remote":
                if not profile.url:
                    raise UnsupportedError(f"Remote agent {profile.name!r} requires a URL")
                _url(profile.url)
                continue
            if (driver := drivers.get(profile.kind)) is None:
                raise UnsupportedError(
                    f"No driver registered for agent {profile.name!r}. Supply an explicit registry."
                )
            local_drivers[profile.name] = driver

        for name, driver in local_drivers.items():
            profile = self._profiles[name]
            fallbacks = tuple(
                (self._profiles[target], local_drivers[target]) for target in fallback_orders[name]
            )
            card = self._cards[name] = build_agent_card(profile)
            self._handlers[name] = _ReservedRequestHandler(
                ledger,
                JohnAgentExecutor(
                    profile, driver, ledger, self._workspaces, self._runtime, fallbacks=fallbacks
                ),
                card,
            )

    @classmethod
    @asynccontextmanager
    async def open(
        cls,
        config_path: str | Path = "agents.toml",
        *,
        drivers: Mapping[str, Driver] | None = None,
    ) -> AsyncIterator[Self]:
        """Load a TOML configuration and close its resources on context exit."""
        profiles, workspaces, db_path = _read_config(Path(config_path).expanduser().resolve())
        db_path.parent.mkdir(parents=True, exist_ok=True)
        ledger = await SqliteLedger.open(db_path)
        try:
            john = cls(profiles=profiles, workspaces=workspaces, ledger=ledger, drivers=drivers)
        except BaseException:
            await ledger.close()
            raise
        async with john:
            yield john

    async def __aenter__(self) -> Self:
        try:
            await self._start()
        except BaseException:
            await self.aclose()
            raise
        return self

    async def __aexit__(self, *_exception: object) -> None:
        await self.aclose()

    async def _start(self) -> None:
        async with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("John is closed")
            if self._started:
                return
            await self.ledger.recover_crashed_runs()
            peers = [profile for profile in self._profiles.values() if profile.kind == "remote"]
            if peers:
                await self._discover(peers)
            self._started = True

    async def _discover(self, profiles: Sequence[AgentProfile]) -> None:
        client = httpx.AsyncClient(
            headers={VERSION_HEADER: PROTOCOL_VERSION_1_0},
            timeout=httpx.Timeout(None, connect=10.0),
            trust_env=False,
        )
        cards: dict[str, AgentCard] = {}
        remotes: dict[str, _RemoteAgent] = {}
        try:
            for profile in profiles:
                assert profile.url is not None
                url = _url(profile.url)
                origin = str(url.copy_with(path="/", query=None, fragment=None))
                card_path = (
                    url.path
                    if url.path.endswith(AGENT_CARD_WELL_KNOWN_PATH)
                    else AGENT_CARD_WELL_KNOWN_PATH
                )
                card = await A2ACardResolver(
                    client, origin, agent_card_path=card_path
                ).get_agent_card(http_kwargs={"timeout": 10.0})
                interface: AgentInterface | None = next(
                    (
                        item
                        for item in card.supported_interfaces
                        if item.protocol_binding == "JSONRPC"
                        and item.protocol_version == PROTOCOL_VERSION_1_0
                    ),
                    None,
                )
                if interface is None:
                    raise UnsupportedError(
                        f"Agent {profile.name!r} has no A2A 1.0 JSON-RPC interface"
                    )
                transport = JsonRpcTransport(
                    client, card, str(_url(interface.url)), is_shared_httpx=True
                )
                cards[profile.name] = card
                remotes[profile.name] = _RemoteAgent(transport, interface.tenant)
        except BaseException:
            await client.aclose()
            raise
        self._http = client
        self._cards.update(cards)
        self._remotes.update(remotes)

    async def aclose(self) -> None:
        async with self._lifecycle_lock:
            if self._closed:
                return
            self._closed = True
            self._direct_responses.clear()
            async with AsyncExitStack() as cleanup:
                cleanup.push_async_callback(self.ledger.close)
                if self._http is not None:
                    cleanup.push_async_callback(self._http.aclose)
                for handler in self._handlers.values():
                    cleanup.push_async_callback(handler.aclose)

    def _profile(self, agent: str) -> AgentProfile:
        try:
            return self._profiles[agent]
        except KeyError:
            raise UnsupportedError(f"Unknown agent {agent!r}") from None

    def is_local(self, agent: str) -> bool:
        """Return whether an agent has an embedded handler and native events."""
        self._profile(agent)
        return agent in self._handlers

    def list_cards(self) -> dict[str, AgentCard]:
        """Return copies of the local and discovered remote agent cards."""
        if len(self._cards) != len(self._profiles):
            raise RuntimeError("Initialize John before listing remote cards")
        return {
            name: AgentCard.FromString(self._cards[name].SerializeToString())
            for name in self._profiles
        }

    async def _parent_run(self, parent: Parent | None) -> str | None:
        if parent is None:
            return None
        if isinstance(parent, str):
            run = await self.ledger.get_run(parent)
            if run is not None and run["owner"] == _LOCAL_OWNER:
                return parent
        else:
            agent, task_id = parent
            for run in reversed(await self.ledger.list_runs(task_id=task_id)):
                if run["owner"] == _LOCAL_OWNER and run["agent_name"] == agent:
                    return run["run_id"]
        raise TaskNotFoundError("No accessible run for the requested parent")

    async def assign(
        self,
        agent: str,
        prompt: str,
        *,
        workspace: Workspace | None = None,
        outputs: Sequence[OutputSpec] | None = None,
        inputs: Sequence[Artifact] | None = None,
        parent: Parent | None = None,
        schema: dict[str, Any] | None = None,
        policy: ApprovalPolicy | None = None,
        on_event: EventSink | None = None,
        message_id: str | None = None,
        timeout_s: float | None = None,
    ) -> Task:
        """Submit a new A2A task and wait for a reply or terminal state."""
        self._profile(agent)
        if outputs and workspace is None:
            raise WorkspaceViolationError("Declared outputs require a workspace")
        artifacts = tuple(inputs or ())
        message = Message(
            message_id=message_id or str(uuid4()),
            role=Role.ROLE_USER,
            parts=[
                Part(text=prompt, media_type="text/plain"),
                *(p for a in artifacts for p in a.parts),
            ],
        )
        if workspace is not None:
            scope = WorkspaceRequest(
                root=workspace.root, access=workspace.access, outputs=outputs or ()
            )
            message.metadata.update(
                {WORKSPACE_EXT_URI: scope.model_dump(mode="json", by_alias=True)}
            )
            message.extensions.append(WORKSPACE_EXT_URI)
        await self._start()
        if agent in self._remotes:
            if any(value is not None for value in (parent, schema, policy, on_event)):
                raise UnsupportedError(
                    "Parent runs, schemas, policies, and native events are local options"
                )
            return await self._send_remote(agent, message, timeout_s)
        call = _call_context(
            message.extensions,
            schema=schema,
            parent_run_id=await self._parent_run(parent),
            parent=parent,
            input_ids=[artifact.artifact_id for artifact in artifacts],
            policy=policy,
            on_event=on_event,
            timeout_s=timeout_s,
        )
        return await self._handlers[agent].on_message_send(
            SendMessageRequest(message=message), call
        )

    async def reply(
        self,
        agent: str,
        task: Task,
        answer: str,
        *,
        message_id: str | None = None,
        policy: ApprovalPolicy | None = None,
        on_event: EventSink | None = None,
        timeout_s: float | None = None,
    ) -> Task:
        """Send a follow-up using the task and context IDs from the saved task."""
        profile = self._profile(agent)
        if not task.id or not task.context_id:
            raise ValueError("A follow-up requires both the task ID and context ID")
        await self._start()
        workspace = profile.require_workspace or _workspace_history(task)
        message = Message(
            message_id=message_id or str(uuid4()),
            task_id=task.id,
            context_id=task.context_id,
            role=Role.ROLE_USER,
            parts=[Part(text=answer, media_type="text/plain")],
            extensions=[WORKSPACE_EXT_URI] if workspace else [],
        )
        if agent in self._remotes:
            if task.id.startswith(_DIRECT_RESPONSE_PREFIX):
                raise UnsupportedError("Remote peer returned a direct message with no task")
            if policy is not None or on_event is not None:
                raise UnsupportedError("Policies and native events are local options")
            return await self._send_remote(agent, message, timeout_s)
        call = _call_context(
            message.extensions, policy=policy, on_event=on_event, timeout_s=timeout_s
        )
        return await self._handlers[agent].on_message_send(
            SendMessageRequest(message=message), call
        )

    async def _send_remote(self, agent: str, message: Message, timeout_s: float | None) -> Task:
        peer = self._remotes[agent]
        extensions = self._cards[agent].capabilities.extensions
        if any(ext.required and ext.uri != WORKSPACE_EXT_URI for ext in extensions):
            raise UnsupportedError("The remote agent requires an unsupported A2A extension")
        workspace = next((ext for ext in extensions if ext.uri == WORKSPACE_EXT_URI), None)
        activated = WORKSPACE_EXT_URI in message.extensions
        if activated and workspace is None:
            raise UnsupportedError("The remote agent does not advertise the workspace extension")
        scope = message.metadata.fields.get(WORKSPACE_EXT_URI)
        if workspace is not None and scope is not None:
            enforcement = workspace.params.fields.get("enforcedSandbox")
            access = scope.struct_value.fields["access"].string_value
            if enforcement is not None and access not in {
                value.string_value for value in enforcement.list_value.values
            }:
                raise UnsupportedError(
                    "The remote agent cannot enforce the requested workspace access"
                )
        if workspace is not None and workspace.required:
            if not activated and not message.task_id:
                raise UnsupportedError("The remote agent requires a workspace")
            if not activated:
                message.extensions.append(WORKSPACE_EXT_URI)
            activated = True
        response = await peer.transport.send_message(
            SendMessageRequest(message=message, tenant=peer.tenant),
            context=_peer_context(workspace=activated, timeout_s=timeout_s),
        )
        if response.HasField("task"):
            return response.task
        if response.HasField("message"):
            answer = response.message
            task = Task(
                id=f"{_DIRECT_RESPONSE_PREFIX}{answer.message_id or message.message_id}",
                context_id=answer.context_id or message.context_id,
                status=TaskStatus(state=TaskState.TASK_STATE_COMPLETED, message=answer),
            )
            key = (agent, task.id)
            self._direct_responses[key] = Task.FromString(task.SerializeToString())
            self._direct_responses.move_to_end(key)
            if len(self._direct_responses) > _MAX_DIRECT_RESPONSES:
                self._direct_responses.popitem(last=False)
            return task
        raise TransportError("The remote agent returned an empty message response")

    def _remote_task_context(self, agent: str) -> ClientCallContext:
        required = any(
            ext.required and ext.uri == WORKSPACE_EXT_URI
            for ext in self._cards[agent].capabilities.extensions
        )
        return _peer_context(workspace=required)

    async def get(self, agent: str, task_id: str) -> Task:
        self._profile(agent)
        await self._start()
        if (peer := self._remotes.get(agent)) is not None:
            if task_id.startswith(_DIRECT_RESPONSE_PREFIX):
                key = (agent, task_id)
                task = self._direct_responses.get(key)
                if task is None:
                    raise TaskNotFoundError("The direct message snapshot is no longer available")
                self._direct_responses.move_to_end(key)
                return Task.FromString(task.SerializeToString())
            return await peer.transport.get_task(
                GetTaskRequest(id=task_id, tenant=peer.tenant),
                context=self._remote_task_context(agent),
            )
        task = await self._handlers[agent].on_get_task(GetTaskRequest(id=task_id), _call_context())
        if task is None:
            raise TaskNotFoundError()
        return task

    async def cancel(self, agent: str, task_id: str) -> Task:
        self._profile(agent)
        await self._start()
        if (peer := self._remotes.get(agent)) is not None:
            if task_id.startswith(_DIRECT_RESPONSE_PREFIX):
                raise UnsupportedError("Remote peer returned a direct message with no task")
            return await peer.transport.cancel_task(
                CancelTaskRequest(id=task_id, tenant=peer.tenant),
                context=self._remote_task_context(agent),
            )
        task = await self._handlers[agent].on_cancel_task(
            CancelTaskRequest(id=task_id), _call_context()
        )
        if task is None:
            raise TaskNotFoundError()
        return task

    def build_http_app(self, agent_name: str, *, url: str | None = None) -> Starlette:
        """Expose an embedded handler over A2A JSON-RPC for the app's lifespan."""
        profile = self._profile(agent_name)
        if (handler := self._handlers.get(agent_name)) is None:
            raise UnsupportedError("Serving requires an embedded agent")
        advertised = url or profile.url or _DEFAULT_SERVE_URL
        interface = _url(advertised)
        require_loopback_host(interface.host)
        path = interface.path or "/"
        card = build_agent_card(profile, url=advertised)

        @asynccontextmanager
        async def lifespan(_app: Starlette) -> AsyncIterator[None]:
            async with self:
                yield

        return Starlette(
            routes=[
                *create_agent_card_routes(card),
                *create_jsonrpc_routes(
                    handler, path, context_builder=DefaultServerCallContextBuilder()
                ),
            ],
            middleware=[Middleware(_LoopbackOnly)],
            lifespan=lifespan,
        )
