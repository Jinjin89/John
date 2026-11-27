"""Project local driver turns into the A2A executor and agent card contracts."""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import AsyncExitStack
from dataclasses import dataclass, replace
from math import isfinite
from pathlib import Path
from typing import Any
from uuid import uuid4

from a2a.helpers.proto_helpers import new_data_part
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.context import ServerCallContext
from a2a.server.events import EventQueue
from a2a.server.events.event_queue_v2 import EventQueueSource
from a2a.server.owner_resolver import OwnerResolver, resolve_user_scope
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentExtension,
    AgentInterface,
    AgentSkill,
    Artifact,
    Message,
    Part,
    Task,
    TaskState,
    TaskStatus,
)
from a2a.utils.constants import TERMINAL_TASK_STATES
from a2a.utils.errors import ExtensionSupportRequiredError, InternalError, InvalidParamsError
from google.protobuf.json_format import MessageToDict, ParseError
from pydantic import ValidationError

from john.ledger import RUN_TERMINAL_STATES, SqliteLedger
from john.runtime import SessionRuntime
from john.transport import redact_text
from john.types import (
    LIFECYCLE_EXT_URI,
    WORKSPACE_EXT_URI,
    WORKSPACE_REF_MEDIA_TYPE,
    AgentProfile,
    ApprovalPolicy,
    AuthRequiredError,
    BusyError,
    Driver,
    FinishedEvent,
    JohnError,
    ResumeUnavailableError,
    SkillSpec,
    ToolEvent,
    TurnFinished,
    TurnNeedsInput,
    TurnOutcome,
    TurnRequest,
    TurnResult,
    UnsupportedError,
    WorkspaceAccess,
    WorkspaceRef,
    WorkspaceRequest,
    WorkspaceViolationError,
    deny_all,
)
from john.workspace import (
    ExtensionRequiredError,
    WorkspaceRegistry,
    assert_outputs_absent,
    build_workspace_artifact,
    parse_workspace_request,
    validate_workspace_request,
    verify_workspace_ref,
)

_LOG = logging.getLogger(__name__)
_CHECKPOINT_EXECUTOR = "__john_executor_agent__"

type _Candidate = tuple[AgentProfile, Driver]


def build_agent_card(profile: AgentProfile, *, url: str = "http://john.local/a2a") -> AgentCard:
    """Publish the configured skill, workspace scope, and streaming capability."""
    description = profile.description or f"Handles requests for {profile.name}."
    skills = profile.skills or (
        SkillSpec(
            id=profile.name,
            name=profile.name,
            description=description,
            input_modes=profile.default_input_modes,
            output_modes=profile.default_output_modes,
        ),
    )
    return AgentCard(
        name=profile.name,
        description=description,
        version=profile.version,
        supported_interfaces=[
            AgentInterface(url=url, protocol_binding="JSONRPC", protocol_version="1.0")
        ],
        capabilities=AgentCapabilities(
            streaming=profile.capabilities.streaming,
            push_notifications=False,
            extensions=[
                AgentExtension(
                    uri=WORKSPACE_EXT_URI,
                    description="Scoped local workspace access and SHA-256 artifact references.",
                    required=profile.require_workspace,
                    params={
                        "enforcedSandbox": sorted(profile.capabilities.enforced_sandbox),
                        "defaultAccess": profile.default_access,
                    },
                )
            ],
        ),
        default_input_modes=profile.default_input_modes,
        default_output_modes=profile.default_output_modes,
        skills=[
            AgentSkill(
                id=skill.id,
                name=skill.name,
                description=skill.description or description,
                tags=skill.tags or (skill.id,),
                input_modes=skill.input_modes,
                output_modes=skill.output_modes,
            )
            for skill in skills
        ],
    )


@dataclass(slots=True)
class _Execution:
    published: asyncio.Future[TaskState.ValueType | None]
    run_id: str | None = None
    cancel_requested: bool = False


@dataclass(frozen=True, slots=True)
class _PreparedTurn:
    request: TurnRequest
    workspace: WorkspaceRequest | None
    workspace_path: Path | None
    parent_run_id: str | None
    resume_ref: str | None
    continuation_run_id: str | None
    candidate: _Candidate


def _remaining(deadline: float | None) -> float | None:
    return None if deadline is None else max(0.0, deadline - asyncio.get_running_loop().time())


def _auth_failed(result: TurnResult) -> bool:
    return result.status == "failed" and result.reason == AuthRequiredError.code


async def _update_status(
    updater: TaskUpdater,
    state: TaskState.ValueType,
    text: str = "",
    *,
    metadata: dict[str, Any] | None = None,
) -> TaskState.ValueType:
    message = (
        updater.new_agent_message([Part(text=text, media_type="text/plain")]) if text else None
    )
    await updater.update_status(state, message=message, metadata=metadata)
    return state


class JohnAgentExecutor(AgentExecutor):
    """Authorize each request and let the SDK persist and publish the task projection."""

    def __init__(
        self,
        profile: AgentProfile,
        driver: Driver,
        ledger: SqliteLedger,
        workspaces: WorkspaceRegistry,
        runtime: SessionRuntime,
        *,
        owner_resolver: OwnerResolver = resolve_user_scope,
        default_policy: ApprovalPolicy = deny_all,
        cancel_timeout_s: float = 10.0,
        fallbacks: tuple[_Candidate, ...] = (),
    ) -> None:
        self._profile = profile
        self._primary = (profile, driver)
        self._fallbacks = fallbacks
        self._candidates = {
            candidate[0].name: candidate for candidate in (self._primary, *fallbacks)
        }
        self._ledger = ledger
        self._workspaces = workspaces
        self._runtime = runtime
        self._owner_resolver = owner_resolver
        self._default_policy = default_policy
        self._cancel_timeout_s = cancel_timeout_s
        self._executions: dict[tuple[str, str], _Execution] = {}

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        task_id, context_id, message = context.task_id, context.context_id, context.message
        if not task_id or not context_id or message is None:
            raise InvalidParamsError("Execution requires a task ID, context ID, and message")
        owner = self._owner_resolver(context.call_context)
        key = (owner, task_id)
        if key in self._executions:
            raise InvalidParamsError("Task already has an active execution")
        execution = _Execution(asyncio.get_running_loop().create_future())
        self._executions[key] = execution
        updater = TaskUpdater(event_queue, task_id, context_id)
        state = None
        try:
            state = await self._execute(context, message, updater, owner, execution)
        finally:
            execution.published.set_result(state)
            del self._executions[key]

    async def _execute(
        self,
        context: RequestContext,
        message: Message,
        updater: TaskUpdater,
        owner: str,
        execution: _Execution,
    ) -> TaskState.ValueType | None:
        started_at = asyncio.get_running_loop().time()
        try:
            prepared = await self._prepare(context, message, updater, owner)
            if execution.cancel_requested:
                return None
            await updater.start_work()
        except Exception as exc:
            return await self._handle_setup_error(context, updater, owner, execution, exc)

        timeout = prepared.request.timeout_s
        deadline = None if timeout is None else started_at + timeout
        replying = context.current_task is not None
        fallbacks = () if replying else self._fallbacks
        attempted_agents: list[str] = []
        parent_run_id = prepared.parent_run_id
        for index, candidate in enumerate((prepared.candidate, *fallbacks)):
            if execution.cancel_requested:
                return None
            if _remaining(deadline) == 0:
                break
            profile, _ = candidate
            turn = prepared
            if index:
                try:
                    path = await asyncio.to_thread(
                        self._prepare_fallback, context, message, owner, prepared, profile
                    )
                except (WorkspaceViolationError, UnsupportedError) as exc:
                    self._log_skipped_fallback(profile, exc)
                    continue
                turn = replace(
                    prepared,
                    request=prepared.request.model_copy(update={"run_id": str(uuid4())}),
                    workspace_path=path,
                    parent_run_id=parent_run_id,
                    candidate=candidate,
                )

            try:
                attempt = await self._run_attempt(
                    context, updater, owner, execution, turn, deadline, fallback=bool(index)
                )
            except Exception as exc:
                return await self._handle_setup_error(context, updater, owner, execution, exc)
            if execution.cancel_requested and execution.run_id is None:
                return None
            if attempt is None:
                continue
            outcome, next_seq = attempt
            attempted_agents.append(profile.name)
            lifecycle = self._fallback_metadata(profile) if replying else {}
            if next_seq is None:
                if not fallbacks:
                    return await self._reject_or_fail(
                        updater, AuthRequiredError(), lifecycle=lifecycle
                    )
                # Journal the setup failure under the public task without sending a native turn.
                execution.run_id = turn.request.run_id
                await self._ledger.start_run(
                    execution.run_id,
                    owner=owner,
                    agent_name=self._profile.name,
                    task_id=updater.task_id,
                    context_id=updater.context_id,
                    parent_run_id=turn.parent_run_id,
                    workspace_root_id=turn.workspace.root if turn.workspace is not None else None,
                    workspace_access=self._workspace_access(turn.workspace),
                )
                next_seq = 0

            if (
                fallbacks
                and isinstance(outcome, TurnFinished)
                and _auth_failed(outcome.result)
                and await self._can_fallback(outcome.result.run_id, turn.workspace, owner, profile)
            ):
                if index < len(fallbacks) or execution.cancel_requested:
                    await self._finish_auth_attempt(outcome.result, next_seq)
                    parent_run_id = outcome.result.run_id
                    execution.run_id = None
                    continue
                lifecycle["attempted_agents"] = attempted_agents

            return await self._publish_attempt(updater, owner, turn, outcome, next_seq, lifecycle)

        if execution.cancel_requested:
            return None
        if attempted_agents:
            return await self._reject_or_fail(
                updater, AuthRequiredError(), lifecycle={"attempted_agents": attempted_agents}
            )
        return await self._handle_setup_error(context, updater, owner, execution, TimeoutError())

    def _workspace_access(self, workspace: WorkspaceRequest | None) -> WorkspaceAccess:
        return workspace.access if workspace is not None else self._profile.default_access

    def _prepare_fallback(
        self,
        context: RequestContext,
        message: Message,
        owner: str,
        prepared: _PreparedTurn,
        profile: AgentProfile,
    ) -> Path | None:
        workspace = prepared.workspace
        if workspace is None and profile.require_workspace:
            raise UnsupportedError("Fallback agent requires a workspace")
        path = self._workspace_path(workspace, owner, profile)
        self._prepare_inputs(
            message, workspace, WORKSPACE_EXT_URI in context.requested_extensions, owner, profile
        )
        return path

    @staticmethod
    def _log_skipped_fallback(profile: AgentProfile, exc: Exception) -> None:
        _LOG.info(
            "Skipped fallback agent %s (%s)", redact_text(profile.name), redact_text(str(exc))
        )

    async def _run_attempt(
        self,
        context: RequestContext,
        updater: TaskUpdater,
        owner: str,
        execution: _Execution,
        prepared: _PreparedTurn,
        deadline: float | None,
        *,
        fallback: bool,
    ) -> tuple[TurnOutcome, int | None] | None:
        profile, driver = prepared.candidate
        workspace = prepared.workspace
        access = self._workspace_access(workspace)
        attempt: tuple[TurnOutcome, int] | None = None
        try:
            async with AsyncExitStack() as stack:
                if execution.cancel_requested:
                    return None
                if _remaining(deadline) == 0:
                    raise TimeoutError()
                try:
                    async with asyncio.timeout_at(deadline):
                        session = await stack.enter_async_context(
                            driver.open(
                                profile,
                                workspace_path=prepared.workspace_path,
                                access=access,
                                resume_ref=prepared.resume_ref,
                            )
                        )
                except AuthRequiredError:
                    return TurnFinished(
                        result=TurnResult(
                            run_id=prepared.request.run_id,
                            status="failed",
                            reason=AuthRequiredError.code,
                        )
                    ), None
                if execution.cancel_requested:
                    return None
                if workspace is not None and access not in session.capabilities.enforced_sandbox:
                    error = UnsupportedError(
                        "Session cannot enforce the requested workspace access"
                    )
                    if fallback:
                        self._log_skipped_fallback(profile, error)
                        return None
                    raise error
                remaining = _remaining(deadline)
                if remaining == 0:
                    raise TimeoutError()
                execution.run_id = prepared.request.run_id
                options = context.call_context.state
                attempt = await self._runtime.execute_turn(
                    session,
                    prepared.request.model_copy(update={"timeout_s": remaining}),
                    owner=owner,
                    agent_name=self._profile.name,
                    task_id=updater.task_id,
                    context_id=updater.context_id,
                    policy=options.get("john_policy", self._default_policy),
                    parent_run_id=prepared.parent_run_id,
                    workspace_root_id=workspace.root if workspace is not None else None,
                    workspace_access=access,
                    continuation_run_id=prepared.continuation_run_id,
                    on_event=options.get("john_on_event"),
                )
        except Exception as exc:
            if attempt is None:
                raise
            outcome, next_seq = attempt
            native_ref = (
                outcome.native_session_ref
                if isinstance(outcome, TurnNeedsInput)
                else outcome.result.native_session_ref
            )
            attempt = (
                TurnFinished(
                    result=TurnResult(
                        run_id=prepared.request.run_id,
                        status="indeterminate",
                        reason=f"session_close_error: {type(exc).__name__}",
                        native_session_ref=native_ref,
                    )
                ),
                next_seq,
            )
        return attempt

    async def _handle_setup_error(
        self,
        context: RequestContext,
        updater: TaskUpdater,
        owner: str,
        execution: _Execution,
        exc: Exception,
    ) -> TaskState.ValueType | None:
        if execution.cancel_requested:
            return None
        if context.current_task is not None and not isinstance(exc, AuthRequiredError):
            checkpoint = await self._ledger.peek_continuation(
                owner, self._profile.name, updater.task_id, updater.context_id
            )
            if checkpoint is not None:
                return await _update_status(
                    updater, TaskState.TASK_STATE_INPUT_REQUIRED, checkpoint["question"]
                )
        if isinstance(exc, ExtensionSupportRequiredError):
            raise exc
        return await self._reject_or_fail(updater, exc)

    async def _can_fallback(
        self,
        run_id: str,
        workspace: WorkspaceRequest | None,
        owner: str,
        profile: AgentProfile,
    ) -> bool:
        events = await self._ledger.get_run_events(run_id)
        if any(isinstance(event, ToolEvent) for event in events):
            return False
        if workspace is not None and workspace.outputs:
            try:
                await asyncio.to_thread(
                    assert_outputs_absent,
                    workspace.root,
                    workspace.outputs,
                    self._workspaces,
                    principal=owner,
                    agent_name=profile.name,
                )
            except (WorkspaceViolationError, UnsupportedError):
                return False
        return True

    async def _finish_auth_attempt(self, result: TurnResult, next_seq: int) -> None:
        await self._ledger.finish_run(
            result.run_id,
            status=result.status,
            reason=result.reason,
            native_session_ref=result.native_session_ref,
            usage=result.usage,
            finished_event=FinishedEvent(
                run_id=result.run_id, seq=next_seq, status=result.status, reason=result.reason
            ),
        )

    def _fallback_metadata(self, profile: AgentProfile) -> dict[str, Any]:
        if profile.name == self._profile.name:
            return {}
        return {"fallback_from": self._profile.name, "resolved_agent": profile.name}

    async def _publish_attempt(
        self,
        updater: TaskUpdater,
        owner: str,
        prepared: _PreparedTurn,
        outcome: TurnOutcome,
        next_seq: int,
        lifecycle: dict[str, Any],
    ) -> TaskState.ValueType:
        run_id = prepared.request.run_id
        try:
            if isinstance(outcome, TurnNeedsInput):
                state = await self._publish_pause(
                    updater, owner, prepared, outcome, next_seq, lifecycle
                )
            else:
                state = await self._publish_terminal(
                    updater, owner, prepared, outcome.result, next_seq, lifecycle
                )
            queue = updater.event_queue
            # The V2 queue can silently drop an enqueue during shutdown.
            if isinstance(queue, EventQueueSource) and queue.is_closed():
                raise InternalError("Task event queue closed during publication")
            return state
        except (Exception, asyncio.CancelledError):
            finished = FinishedEvent(
                run_id=run_id,
                seq=next_seq,
                status="indeterminate",
                reason="publish_failed",
            )
            try:
                await self._ledger.finish_run(
                    run_id,
                    status=finished.status,
                    reason=finished.reason,
                    finished_event=finished,
                    project_failure=True,
                )
            except Exception as exc:
                _LOG.error("Could not record result publication failure (%s)", type(exc).__name__)
            raise

    async def validate_reply(self, task: Task, message: Message, call: ServerCallContext) -> None:
        """Validate a new follow-up before the SDK begins changing its task."""
        try:
            owner = self._owner_resolver(call)
            _, workspace, (profile, _) = await self._resume_scope(task, message, call, owner)
            self._workspace_path(workspace, owner, profile)
        except ValidationError:
            raise InvalidParamsError("Invalid saved workspace scope") from None
        except (WorkspaceViolationError, UnsupportedError, ResumeUnavailableError) as exc:
            raise InvalidParamsError(redact_text(str(exc))) from None

    async def _resume_scope(
        self,
        task: Task,
        message: Message,
        call: ServerCallContext,
        owner: str,
    ) -> tuple[dict[str, Any], WorkspaceRequest | None, _Candidate]:
        checkpoint = await self._ledger.peek_continuation(
            owner, self._profile.name, task.id, task.context_id
        )
        if checkpoint is None:
            raise ResumeUnavailableError("No active continuation for this task")
        continuation = checkpoint["continuation"]
        candidate = self._primary
        lifecycle = task.metadata.fields.get(LIFECYCLE_EXT_URI)
        executor = lifecycle.struct_value.fields.get("resolved_agent") if lifecycle else None
        if executor is not None:
            name = executor.string_value
            if (
                name not in self._candidates
                or continuation.get(_CHECKPOINT_EXECUTOR) != name
                or not isinstance(continuation.get("continuation"), dict)
            ):
                raise ResumeUnavailableError("The executor for this continuation is unavailable")
            candidate = self._candidates[name]
            checkpoint["continuation"] = continuation["continuation"]
        profile, _ = candidate
        workspace = (
            WorkspaceRequest.model_validate(checkpoint["workspace_req"])
            if checkpoint["workspace_req"] is not None
            else None
        )
        if (
            profile.require_workspace or self._profile.require_workspace or workspace is not None
        ) and (WORKSPACE_EXT_URI not in call.requested_extensions):
            raise ExtensionSupportRequiredError("Activate the required workspace extension")
        if workspace is None and profile.require_workspace:
            raise WorkspaceViolationError("The saved executor requires a workspace")
        metadata = MessageToDict(message.metadata)
        if WORKSPACE_EXT_URI in metadata:
            supplied = parse_workspace_request(metadata, call.requested_extensions, required=False)
            if supplied != workspace:
                raise WorkspaceViolationError("A follow-up must use its saved workspace scope")
        return checkpoint, workspace, candidate

    def _workspace_path(
        self, workspace: WorkspaceRequest | None, owner: str, profile: AgentProfile
    ) -> Path | None:
        if workspace is None:
            return None
        return validate_workspace_request(
            workspace,
            self._workspaces,
            profile.capabilities,
            principal=owner,
            agent_name=profile.name,
        ).host_path

    async def _prepare(
        self,
        context: RequestContext,
        message: Message,
        updater: TaskUpdater,
        owner: str,
    ) -> _PreparedTurn:
        options = context.call_context.state
        workspace = None
        schema = options.get("john_schema")
        parent_run_id = options.get("john_parent_run_id")
        continuation = resume_ref = continuation_run_id = None
        candidate = self._primary
        if context.current_task is None:
            workspace_error = None
            try:
                workspace = parse_workspace_request(
                    MessageToDict(message.metadata),
                    context.requested_extensions,
                    required=self._profile.require_workspace,
                )
            except ExtensionRequiredError as exc:
                raise ExtensionSupportRequiredError(str(exc)) from None
            except WorkspaceViolationError as exc:
                workspace_error = exc
            await updater.event_queue.enqueue_event(
                Task(
                    id=updater.task_id,
                    context_id=updater.context_id,
                    status=TaskStatus(state=TaskState.TASK_STATE_SUBMITTED),
                    history=[message],
                )
            )
            if workspace_error is not None:
                raise workspace_error
        else:
            checkpoint, workspace, candidate = await self._resume_scope(
                context.current_task, message, context.call_context, owner
            )
            schema = checkpoint["schema_def"]
            continuation = checkpoint["continuation"]
            resume_ref = checkpoint["native_session_ref"]
            parent_run_id = continuation_run_id = checkpoint["run_id"]

        profile, _ = candidate
        path = self._workspace_path(workspace, owner, profile)
        prompt, refs = await asyncio.to_thread(
            self._prepare_inputs,
            message,
            workspace,
            WORKSPACE_EXT_URI in context.requested_extensions,
            owner,
            profile,
        )
        request = TurnRequest(
            run_id=str(uuid4()),
            prompt=prompt,
            input_refs=refs,
            outputs=workspace.outputs if workspace is not None else (),
            schema_def=schema,
            continuation=continuation,
            timeout_s=options.get("john_timeout_s"),
        )
        if request.timeout_s is not None and (
            not isfinite(request.timeout_s) or request.timeout_s < 0
        ):
            raise InvalidParamsError("Execution timeout must be finite and non-negative")
        return _PreparedTurn(
            request, workspace, path, parent_run_id, resume_ref, continuation_run_id, candidate
        )

    def _prepare_inputs(
        self,
        message: Message,
        workspace: WorkspaceRequest | None,
        extension_active: bool,
        owner: str,
        profile: AgentProfile,
    ) -> tuple[str, tuple[WorkspaceRef, ...]]:
        accepts_json = "application/json" in profile.default_input_modes or any(
            "application/json" in skill.input_modes for skill in profile.skills
        )
        prompt = []
        refs = []
        for part in message.parts:
            content = part.WhichOneof("content")
            if part.media_type != WORKSPACE_REF_MEDIA_TYPE:
                if content == "text":
                    prompt.append(part.text)
                elif content == "data" and part.media_type == "application/json" and accepts_json:
                    prompt.append(
                        json.dumps(
                            MessageToDict(part.data),
                            sort_keys=True,
                            separators=(",", ":"),
                            ensure_ascii=False,
                            allow_nan=False,
                        )
                    )
                else:
                    raise UnsupportedError("Input part is not supported by this agent")
                continue
            if workspace is None or not extension_active:
                raise WorkspaceViolationError(
                    "Workspace references require an active workspace scope"
                )
            if content != "data":
                raise WorkspaceViolationError("Expected a workspace reference data part")
            data = MessageToDict(part.data)
            if not isinstance(data, dict) or type(
                data.get("sizeBytes", data.get("size_bytes"))
            ) not in (int, float):
                raise WorkspaceViolationError("Expected a workspace reference with a numeric size")
            try:
                ref = WorkspaceRef.model_validate(data)
            except ValidationError:
                raise WorkspaceViolationError("Invalid workspace reference data") from None
            if ref.root != workspace.root:
                raise WorkspaceViolationError("Workspace reference is outside the requested root")
            refs.append(
                verify_workspace_ref(
                    ref, self._workspaces, principal=owner, agent_name=profile.name
                )
            )
        if workspace is not None and workspace.outputs:
            assert_outputs_absent(
                workspace.root,
                workspace.outputs,
                self._workspaces,
                principal=owner,
                agent_name=profile.name,
            )
        return "\n".join(prompt), tuple(refs)

    async def _reject_or_fail(
        self, updater: TaskUpdater, exc: Exception, *, lifecycle: dict[str, Any] | None = None
    ) -> TaskState.ValueType:
        if isinstance(exc, AuthRequiredError):
            return await _update_status(
                updater,
                TaskState.TASK_STATE_FAILED,
                AuthRequiredError.code,
                metadata={LIFECYCLE_EXT_URI: {**(lifecycle or {}), "auth_required": True}},
            )
        if isinstance(exc, TimeoutError):
            return await _update_status(updater, TaskState.TASK_STATE_FAILED, "turn_timeout")
        if isinstance(
            exc, WorkspaceViolationError | UnsupportedError | BusyError | InvalidParamsError
        ):
            return await _update_status(
                updater, TaskState.TASK_STATE_REJECTED, redact_text(str(exc))
            )
        if isinstance(exc, ValidationError):
            return await _update_status(
                updater, TaskState.TASK_STATE_REJECTED, "Invalid turn parameters"
            )
        if isinstance(exc, ResumeUnavailableError):
            return await _update_status(updater, TaskState.TASK_STATE_FAILED, redact_text(str(exc)))
        reason = (
            exc.code if isinstance(exc, JohnError) else f"execution_error: {type(exc).__name__}"
        )
        return await _update_status(updater, TaskState.TASK_STATE_FAILED, reason)

    async def _publish_pause(
        self,
        updater: TaskUpdater,
        owner: str,
        prepared: _PreparedTurn,
        outcome: TurnNeedsInput,
        next_seq: int,
        lifecycle: dict[str, Any],
    ) -> TaskState.ValueType:
        finished = FinishedEvent(
            run_id=outcome.run_id, seq=next_seq, status="incomplete", reason="input_required"
        )
        profile, _ = prepared.candidate
        continuation = outcome.continuation
        if profile.name != self._profile.name:
            continuation = {_CHECKPOINT_EXECUTOR: profile.name, "continuation": continuation}
        try:
            await self._ledger.stage_paused_run(
                updater.task_id,
                owner=owner,
                agent_name=self._profile.name,
                context_id=updater.context_id,
                run_id=outcome.run_id,
                question=outcome.question,
                continuation=continuation,
                workspace_req=prepared.workspace,
                schema_def=prepared.request.schema_def,
                native_session_ref=outcome.native_session_ref,
                finished_event=finished,
            )
        except (TypeError, ValueError):
            failure = TurnResult(
                run_id=outcome.run_id,
                status="failed",
                reason="invalid_continuation",
                native_session_ref=outcome.native_session_ref,
            )
            return await self._publish_terminal(
                updater, owner, prepared, failure, next_seq, lifecycle
            )
        lifecycle.update(self._fallback_metadata(profile))
        return await _update_status(
            updater,
            TaskState.TASK_STATE_INPUT_REQUIRED,
            outcome.question,
            metadata={LIFECYCLE_EXT_URI: lifecycle} if lifecycle else None,
        )

    def _artifacts(
        self, result: TurnResult, workspace: WorkspaceRequest | None, owner: str, agent_name: str
    ) -> list[Artifact]:
        if workspace is not None and workspace.outputs:
            return [
                build_workspace_artifact(
                    workspace.root,
                    spec,
                    self._workspaces,
                    principal=owner,
                    agent_name=agent_name,
                )
                for spec in workspace.outputs
            ]
        if result.has_structured:
            part = new_data_part(result.structured, media_type="application/json")
        elif result.text:
            part = Part(text=result.text, media_type="text/plain")
        else:
            return []
        return [Artifact(artifact_id=str(uuid4()), parts=[part])]

    async def _publish_terminal(
        self,
        updater: TaskUpdater,
        owner: str,
        prepared: _PreparedTurn,
        result: TurnResult,
        next_seq: int,
        lifecycle: dict[str, Any],
    ) -> TaskState.ValueType:
        artifacts = []
        if result.status == "completed":
            try:
                artifacts = await asyncio.to_thread(
                    self._artifacts, result, prepared.workspace, owner, prepared.candidate[0].name
                )
            except (WorkspaceViolationError, UnsupportedError) as exc:
                result = result.model_copy(
                    update={"status": "failed", "reason": f"invalid_output: {exc}"}
                )
            except (ParseError, TypeError, ValueError, OverflowError):
                result = result.model_copy(update={"status": "failed", "reason": "invalid_output"})

        reason = redact_text(result.reason)
        await self._ledger.stage_terminal_run(
            updater.task_id,
            owner=owner,
            agent_name=self._profile.name,
            run_id=result.run_id,
            status=result.status,
            reason=reason,
            native_session_ref=result.native_session_ref,
            usage=result.usage,
            finished_event=FinishedEvent(
                run_id=result.run_id, seq=next_seq, status=result.status, reason=reason
            ),
            artifacts=artifacts,
        )
        if result.status == "indeterminate":
            lifecycle["indeterminate"] = True
        if result.status == "completed":
            lifecycle.update(self._fallback_metadata(prepared.candidate[0]))
        if _auth_failed(result):
            lifecycle["auth_required"] = True
        return await _update_status(
            updater,
            RUN_TERMINAL_STATES[result.status],
            reason,
            metadata={LIFECYCLE_EXT_URI: lifecycle} if lifecycle else None,
        )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        task_id, context_id = context.task_id, context.context_id
        if not task_id or not context_id:
            raise InvalidParamsError("Cancellation requires a task ID and context ID")
        owner = self._owner_resolver(context.call_context)
        execution = self._executions.get((owner, task_id))
        if execution is not None:
            execution.cancel_requested = True
            if (run_id := execution.run_id) is not None:
                await self._runtime.cancel_task(
                    task_id,
                    self._cancel_timeout_s,
                    owner=owner,
                    agent_name=self._profile.name,
                    run_id=run_id,
                )
                # V2 stops its producer when this hook returns.
                published = await asyncio.shield(execution.published)
                if published in TERMINAL_TASK_STATES:
                    return
                if published is None and execution.run_id is not None:
                    raise InternalError("Execution ended before publishing a cancellation outcome")
        await TaskUpdater(event_queue, task_id, context_id).cancel()
