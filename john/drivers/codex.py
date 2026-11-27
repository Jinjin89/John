"""Codex app-server 0.172 stdio driver."""

from __future__ import annotations

import asyncio
import json
import os
import re
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from itertools import count
from pathlib import Path
from typing import Any

from john.transport import (
    Frame,
    JsonRpcError,
    JsonRpcPeer,
    MethodNotFound,
    RequestId,
    RpcParams,
    redact_text,
)
from john.types import (
    AgentProfile,
    ApprovalDecision,
    ApprovalPolicy,
    ApprovalRequest,
    AuthRequiredError,
    BusyError,
    Capabilities,
    Driver,
    DriverSession,
    ErrorEvent,
    EventSink,
    FinishedEvent,
    OtherEvent,
    PlanEvent,
    ResumeUnavailableError,
    RuntimeEvent,
    StartedEvent,
    TextEvent,
    ThoughtEvent,
    ToolEvent,
    TransportError,
    TurnFinished,
    TurnRequest,
    TurnResult,
    TurnStatus,
    UnsupportedError,
    UsageEvent,
    WorkspaceAccess,
    WorkspaceViolationError,
)

_RPC_TIMEOUT = 15.0
_SANDBOX_MODE = {"read_only": "read-only", "read_write": "workspace-write"}
_APPROVAL_METHODS = {"item/commandExecution/requestApproval", "item/fileChange/requestApproval"}
_REASONING_DELTAS = {"item/reasoning/summaryTextDelta", "item/reasoning/textDelta"}
_TOOL_ITEMS = {
    "commandExecution",
    "fileChange",
    "mcpToolCall",
    "dynamicToolCall",
    "collabAgentToolCall",
    "functionCallOutput",
    "webSearch",
    "imageView",
    "imageGeneration",
}
_LIFECYCLE_NOTICES = {
    "thread/started",
    "thread/status/changed",
    "turn/started",
    "serverRequest/resolved",
    "remoteControl/status/changed",
    "item/reasoning/summaryPartAdded",
}
_TURN_NOTICES = {
    "turn/completed",
    "item/agentMessage/delta",
    "item/started",
    "item/completed",
    "turn/plan/updated",
    "thread/tokenUsage/updated",
    "error",
    *_REASONING_DELTAS,
}
_AUTH_ERROR = re.compile(r"\b401\b|\bunauthorized\b|\bnot authenticated\b", re.IGNORECASE)


def _object(value: Any) -> Frame:
    return value if isinstance(value, dict) else {}


def _string(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _redact(value: Any) -> Any:
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, list):
        return [_redact(item) for item in value]
    if isinstance(value, dict):
        cleaned = {}
        for key, item in value.items():
            # Apply the transport's credential-key patterns to structured fields.
            probe = f"{key}=value"
            cleaned[redact_text(key)] = (
                "<redacted>" if redact_text(probe) != probe else _redact(item)
            )
        return cleaned
    return value


def _error_details(error: Frame) -> tuple[str, str]:
    message = _string(error.get("message")) or "Codex reported an error"
    info = error.get("codexErrorInfo")
    code = _string(info) or next(iter(_object(info)), "codex_error")
    status = _object(_object(info).get(code)).get("httpStatusCode")
    if code == "unauthorized" or status == 401 or _AUTH_ERROR.search(message):
        code = AuthRequiredError.code
    return redact_text(code), redact_text(message)


def _prompt(request: TurnRequest) -> str:
    sections = [request.prompt]
    if request.input_refs:
        rows = [
            f"- {json.dumps(ref.path, ensure_ascii=False)} ({ref.media_type}, "
            f"{ref.size_bytes} bytes, SHA-256 {ref.sha256})"
            for ref in request.input_refs
        ]
        sections.append("Workspace inputs (paths relative to the workspace):\n" + "\n".join(rows))
    if request.outputs:
        rows = [
            f"- {json.dumps(spec.name, ensure_ascii=False)}: "
            f"{json.dumps(spec.path, ensure_ascii=False)} ({spec.media_type})"
            for spec in request.outputs
        ]
        sections.append(
            "Write the declared workspace outputs at these relative paths:\n" + "\n".join(rows)
        )
    return "\n\n".join(sections)


@dataclass(slots=True)
class _MessageItem:
    deltas: list[str] = field(default_factory=list)
    final: str | None = None
    phase: str | None = None
    completed: bool = False

    @property
    def text(self) -> str:
        return self.final if self.final is not None else "".join(self.deltas)


@dataclass(slots=True)
class _Turn:
    run_id: str
    on_event: EventSink
    on_approval: ApprovalPolicy
    notices: asyncio.Queue[tuple[str, Frame] | None] = field(default_factory=asyncio.Queue)
    bound: asyncio.Event = field(default_factory=asyncio.Event)
    approvals_closed: asyncio.Event = field(default_factory=asyncio.Event)
    seq: count = field(default_factory=count)
    messages: dict[str, _MessageItem] = field(default_factory=dict)
    terminal_ids: set[str] = field(default_factory=set)
    usage: dict[str, int] | None = None
    error_code: str | None = None
    auth_required: bool = False
    sent: bool = False
    settled: bool = False
    cancelled: bool = False
    turn_id: str | None = None
    interrupt: asyncio.Task[None] | None = None
    disconnect_reason: str = "codex_disconnected_before_turn_completed"

    async def emit(self, event: Callable[..., RuntimeEvent], **fields: Any) -> None:
        await self.on_event(event(run_id=self.run_id, seq=next(self.seq), **fields))

    def stop_approvals(self) -> None:
        self.approvals_closed.set()
        self.bound.set()

    @property
    def terminal_received(self) -> bool:
        return self.turn_id is not None and self.turn_id in self.terminal_ids

    @property
    def text(self) -> str:
        final = [item.text for item in self.messages.values() if item.phase == "final_answer"]
        parts = final or [item.text for item in self.messages.values()]
        return redact_text("\n".join(filter(None, parts)))


class CodexAppServerSession(DriverSession):
    def __init__(
        self,
        profile: AgentProfile,
        workspace_path: Path | None,
        access: WorkspaceAccess,
        policy: str,
        ephemeral: bool,
    ) -> None:
        self._workspace_path = workspace_path
        self._access = access
        self._policy = policy
        self._ephemeral = ephemeral
        self._include_raw = bool(profile.options.get("include_raw_other"))
        self._capabilities = Capabilities(
            approvals="live" if policy == "on-request" else "none",
            thinking="stream",
            resume=profile.capabilities.resume and not ephemeral,
            native_schema=True,
            enforced_sandbox=frozenset({access}),
        )
        self._thread_id: str | None = None
        self._last_turn_id: str | None = None
        self._active: _Turn | None = None
        self._peer: JsonRpcPeer

    @property
    def native_ref(self) -> str | None:
        return self._thread_id

    @property
    def capabilities(self) -> Capabilities:
        return self._capabilities

    @property
    def ignored_fields(self) -> dict[str, str]:
        return {}

    async def _initialize(self, cwd: Path, resume_ref: str | None, preflight_auth: bool) -> None:
        await self._setup_request(
            "initialize", {"clientInfo": {"name": "john", "version": "0.1.0"}}
        )
        await self._peer.notify("initialized")
        if preflight_auth:
            state = await self._setup_request("account/read", {"refreshToken": False})
            requires_auth = state.get("requiresOpenaiAuth")
            if type(requires_auth) is not bool:
                raise TransportError("Invalid Codex account/read response")
            if requires_auth and not state.get("account"):
                raise AuthRequiredError("Codex account authentication is required")

        settings: Frame = {
            "cwd": str(cwd),
            "sandbox": _SANDBOX_MODE[self._access],
            "approvalPolicy": self._policy,
            "approvalsReviewer": "user",
        }
        if self._access == "read_write":
            settings["config"] = {
                "sandbox_workspace_write": {
                    "writable_roots": [],
                    "exclude_tmpdir_env_var": True,
                    "exclude_slash_tmp": True,
                    "network_access": False,
                }
            }
        if resume_ref is None:
            response = await self._setup_request(
                "thread/start", {**settings, "ephemeral": self._ephemeral}
            )
        else:
            response = await self._setup_request(
                "thread/resume", {**settings, "threadId": resume_ref, "excludeTurns": True}
            )
        self._verify_scope(response, cwd)
        thread = _object(response.get("thread"))
        thread_id = _string(thread.get("id"))
        if not thread_id:
            raise TransportError("Codex did not return a thread ID")
        if resume_ref is not None and (
            thread_id != resume_ref or _object(thread.get("status")).get("type") == "active"
        ):
            raise ResumeUnavailableError("Codex did not resume the requested idle thread")
        if thread.get("ephemeral") is not self._ephemeral:
            raise UnsupportedError("Codex returned a different thread persistence setting")
        self._thread_id = thread_id

    async def _setup_request(self, method: str, params: Frame) -> Frame:
        try:
            response = await self._peer.request(method, params, timeout_s=_RPC_TIMEOUT)
        except JsonRpcError as exc:
            if method == "thread/resume":
                raise ResumeUnavailableError(
                    f"Codex thread/resume failed (RPC {exc.code})"
                ) from None
            if exc.code == 401 or _AUTH_ERROR.search(exc.message):
                raise AuthRequiredError("Codex account authentication is required") from None
            raise TransportError(f"Codex {method} failed (RPC {exc.code})") from None
        except TimeoutError:
            raise TransportError(f"Codex {method} timed out") from None
        if not isinstance(response, dict):
            raise TransportError(f"Codex {method} returned an invalid result")
        return response

    def _verify_scope(self, response: Frame, cwd: Path) -> None:
        sandbox = _object(response.get("sandbox"))
        expected = "readOnly" if self._access == "read_only" else "workspaceWrite"
        if sandbox.get("type") != expected or sandbox.get("networkAccess", False) is not False:
            raise UnsupportedError("Codex returned a different sandbox policy")
        if response.get("cwd") != str(cwd):
            raise UnsupportedError("Codex returned a different workspace directory")
        if (
            response.get("approvalPolicy") != self._policy
            or response.get("approvalsReviewer") != "user"
        ):
            raise UnsupportedError("Codex returned a different approval policy or reviewer")
        if self._access == "read_write":
            roots = sandbox.get("writableRoots", [])
            if (
                sandbox.get("excludeTmpdirEnvVar") is not True
                or sandbox.get("excludeSlashTmp") is not True
                or not isinstance(roots, list)
                or any(not _within_workspace(root, cwd) for root in roots)
            ):
                raise UnsupportedError("Codex workspace-write includes unscoped writable paths")

    async def run_turn(
        self, request: TurnRequest, on_event: EventSink, on_approval: ApprovalPolicy
    ) -> TurnFinished:
        if self._active is not None:
            raise BusyError("The Codex session already has an active turn")
        if self._peer.transport.closed:
            raise TransportError("The Codex app-server session is closed")
        if request.continuation is not None:
            raise UnsupportedError("Codex sessions resume by native thread ID")
        if request.outputs and self._access != "read_write":
            raise WorkspaceViolationError("Declared file outputs require read-write access")
        if (request.input_refs or request.outputs) and self._workspace_path is None:
            raise WorkspaceViolationError("File inputs and outputs require a workspace binding")
        for spec in (*request.input_refs, *request.outputs):
            if "\x00" in spec.path or any(part in ("", ".", "..") for part in spec.path.split("/")):
                raise WorkspaceViolationError("Expected a workspace-relative POSIX path")

        active = _Turn(request.run_id, on_event, on_approval)
        self._active = active
        try:
            await active.emit(
                StartedEvent, native_session_ref=self.native_ref, ignored_fields=self.ignored_fields
            )
            status, reason = await self._execute(active, request)
            await active.emit(FinishedEvent, status=status, reason=reason)
            return TurnFinished(
                result=TurnResult(
                    run_id=request.run_id,
                    status=status,
                    reason=reason,
                    text=active.text,
                    native_session_ref=self.native_ref,
                    usage=active.usage,
                )
            )
        finally:
            active.stop_approvals()
            if active.turn_id is not None:
                self._last_turn_id = active.turn_id
            try:
                if active.sent and not active.settled:
                    await self._peer.close()
            finally:
                try:
                    if active.interrupt is not None:
                        active.interrupt.cancel()
                        await asyncio.gather(active.interrupt, return_exceptions=True)
                finally:
                    self._active = None

    async def _execute(self, active: _Turn, request: TurnRequest) -> tuple[TurnStatus, str]:
        if active.cancelled:
            return "cancelled", "cancelled_by_caller"
        params = {
            "threadId": self.native_ref,
            "input": [{"type": "text", "text": _prompt(request)}],
        }
        if request.schema_def is not None:
            params["outputSchema"] = request.schema_def
        active.sent = True
        try:
            ack = await self._peer.request("turn/start", params, timeout_s=_RPC_TIMEOUT)
        except JsonRpcError as exc:
            if active.turn_id is not None:
                return "indeterminate", "codex_turn_start_conflicted_with_notifications"
            active.settled = True
            code = (
                AuthRequiredError.code
                if exc.code == 401 or _AUTH_ERROR.search(exc.message)
                else "codex_turn_start_rejected"
            )
            await active.emit(ErrorEvent, code=code, message=exc.message)
            return "failed", code
        except (TransportError, TimeoutError) as exc:
            if isinstance(exc, TimeoutError):
                active.disconnect_reason = "codex_turn_start_acknowledgment_timeout"
            # EOF wakes the RPC waiter before the peer's notification queue is drained.
            # Closing joins that queue before an interrupted dispatch is classified.
            await self._peer.close()
        else:
            turn_id = _string(_object(_object(ack).get("turn")).get("id"))
            if (
                not turn_id
                or turn_id == self._last_turn_id
                or active.turn_id not in (None, turn_id)
            ):
                return "indeterminate", "invalid_codex_turn_start_reply"
            self._bind_turn(active, turn_id)
        return await self._consume(active)

    async def _consume(self, active: _Turn) -> tuple[TurnStatus, str]:
        drained = False
        while not drained or not active.notices.empty():
            notice = await active.notices.get()
            if notice is None:
                await self._peer.close()
                drained = True
                continue
            method, params = notice
            native_id = self._notice_turn_id(method, params)
            if method in _TURN_NOTICES and (
                params.get("threadId") != self.native_ref or native_id is None
            ):
                continue
            if native_id is not None and native_id != active.turn_id:
                continue
            if method == "turn/completed":
                return await self._completed(active, _object(params.get("turn")))
            await self._emit_notice(active, method, params)
        return "indeterminate", active.disconnect_reason

    @staticmethod
    def _notice_turn_id(method: str, params: Frame) -> str | None:
        if method in ("turn/started", "turn/completed"):
            return _string(_object(params.get("turn")).get("id"))
        return _string(params.get("turnId"))

    async def _on_notification(self, method: str, params: RpcParams) -> None:
        active = self._active
        if active is None or not active.sent:
            return
        payload = _object(params)
        thread_id = payload.get("threadId")
        if thread_id is not None and thread_id != self.native_ref:
            return
        turn_id = self._notice_turn_id(method, payload)
        if thread_id == self.native_ref and turn_id:
            if (
                method == "turn/started"
                and active.turn_id is None
                and turn_id != self._last_turn_id
            ):
                self._bind_turn(active, turn_id)
            if method == "turn/completed":
                active.terminal_ids.add(turn_id)
                if active.terminal_received:
                    active.stop_approvals()
        if method not in _LIFECYCLE_NOTICES:
            active.notices.put_nowait((method, payload))

    async def _on_unknown_frame(self, frame: Frame) -> None:
        active = self._active
        if active is not None and active.sent:
            active.notices.put_nowait(("jsonrpc/unknownFrame", frame))

    async def _on_eof(self, exit_code: int | None) -> None:
        active = self._active
        if active is not None:
            active.stop_approvals()
            active.notices.put_nowait(None)

    def _bind_turn(self, active: _Turn, turn_id: str) -> None:
        active.turn_id = turn_id
        active.bound.set()
        if active.terminal_received:
            active.stop_approvals()
        elif active.cancelled:
            self._schedule_interrupt(active)

    async def cancel(self, run_id: str) -> None:
        active = self._active
        if active is None or active.run_id != run_id or active.settled or active.terminal_received:
            return
        active.cancelled = True
        active.stop_approvals()
        self._schedule_interrupt(active)
        if (interrupt := active.interrupt) is not None:
            # The matching terminal event can retire the RPC before its reply.
            await asyncio.wait((interrupt,))
            if not interrupt.cancelled():
                interrupt.result()

    def _schedule_interrupt(self, active: _Turn) -> None:
        if active.turn_id is not None and not active.terminal_received and active.interrupt is None:
            active.interrupt = asyncio.create_task(
                self._interrupt(active), name="Codex turn/interrupt"
            )

    async def _interrupt(self, active: _Turn) -> None:
        try:
            await self._peer.request(
                "turn/interrupt",
                {"threadId": self.native_ref, "turnId": active.turn_id},
                timeout_s=_RPC_TIMEOUT,
            )
        except (JsonRpcError, TransportError, TimeoutError):
            if not active.terminal_received:
                active.disconnect_reason = "codex_interrupt_unconfirmed"
                await self._peer.close()

    async def _on_request(self, method: str, params: RpcParams, request_id: RequestId) -> Frame:
        if method not in _APPROVAL_METHODS:
            raise MethodNotFound
        active = self._active
        payload = _object(params)
        turn_id = _string(payload.get("turnId"))
        if not turn_id or not _string(payload.get("itemId")):
            raise JsonRpcError(-32602, "Invalid Codex approval parameters")
        if (
            active is None
            or not active.sent
            or self._policy == "never"
            or payload.get("threadId") != self.native_ref
        ):
            return {"decision": "decline"}
        await active.bound.wait()
        if active.approvals_closed.is_set():
            return {"decision": "cancel"}
        if turn_id != active.turn_id:
            return {"decision": "decline"}
        args = _redact(payload)
        id_type = "int" if type(request_id) is int else "str"
        request = ApprovalRequest(
            request_id=f"{id_type}:{request_id}",
            run_id=active.run_id,
            action=method,
            summary=_string(args.get("reason")) or _string(args.get("command")) or method,
            args=args,
            offered_options=("accept", "decline", "cancel"),
        )
        return {"decision": await _decide_approval(active, request)}

    async def _emit_notice(self, active: _Turn, method: str, params: Frame) -> None:
        if method == "item/agentMessage/delta":
            item_id, delta = _string(params.get("itemId")), _string(params.get("delta"))
            if item_id and delta:
                item = active.messages.setdefault(item_id, _MessageItem())
                if not item.completed:
                    item.deltas.append(delta)
                    await active.emit(TextEvent, text=redact_text(delta), is_delta=True)
        elif method in _REASONING_DELTAS:
            if delta := _string(params.get("delta")):
                await active.emit(ThoughtEvent, text=redact_text(delta), is_delta=True)
        elif method in ("item/started", "item/completed"):
            await self._item_event(active, _object(params.get("item")), method == "item/completed")
        elif method == "turn/plan/updated":
            plan = params.get("plan")
            if isinstance(plan, list):
                steps = tuple(
                    redact_text(step) for row in plan if (step := _string(_object(row).get("step")))
                )
                await active.emit(PlanEvent, items=steps)
        elif method == "thread/tokenUsage/updated":
            usage = _object(_object(params.get("tokenUsage")).get("last"))
            counts = {
                field: value
                for field, key in (
                    ("input_tokens", "inputTokens"),
                    ("output_tokens", "outputTokens"),
                )
                if type(value := usage.get(key)) is int and value >= 0
            }
            if counts:
                active.usage = counts
                await active.emit(UsageEvent, **counts)
        elif method == "error":
            await self._emit_error(
                active, _object(params.get("error")), bool(params.get("willRetry"))
            )
        else:
            native_type = (
                (_string(params.get("method")) or method)
                if method == "jsonrpc/unknownFrame"
                else method
            )
            await active.emit(
                OtherEvent,
                native_type=redact_text(native_type),
                raw=_redact(params) if self._include_raw else None,
                redacted=True,
            )

    @staticmethod
    async def _item_event(active: _Turn, item: Frame, completed: bool) -> None:
        kind, item_id = _string(item.get("type")), _string(item.get("id"))
        if not item_id:
            return
        if kind == "agentMessage":
            message = active.messages.setdefault(item_id, _MessageItem())
            message.phase = _string(item.get("phase")) or message.phase
            if completed and not message.completed:
                message.completed = True
                message.final = _string(item.get("text"))
                if message.final and not message.deltas:
                    await active.emit(TextEvent, text=redact_text(message.final), is_delta=False)
        elif kind in _TOOL_ITEMS:
            failed = item.get("status") in ("failed", "declined", "interrupted")
            phase = "started" if not completed else ("failed" if failed else "completed")
            summary = _string(item.get("command")) or _string(item.get("tool")) or ""
            await active.emit(
                ToolEvent, call_id=item_id, name=kind, phase=phase, summary=redact_text(summary)
            )

    @staticmethod
    async def _emit_error(active: _Turn, error: Frame, retryable: bool) -> None:
        code, message = _error_details(error)
        active.error_code = code
        active.auth_required |= code == AuthRequiredError.code
        await active.emit(ErrorEvent, code=code, message=message, retryable=retryable)

    async def _completed(self, active: _Turn, turn: Frame) -> tuple[TurnStatus, str]:
        status = turn.get("status")
        if status not in ("completed", "failed", "interrupted"):
            return "indeterminate", "invalid_codex_turn_completion"
        active.settled = True
        active.stop_approvals()
        items = turn.get("items")
        if isinstance(items, list):
            for item in items:
                if _object(item).get("type") == "agentMessage":
                    await self._item_event(active, item, completed=True)
        if status == "completed":
            return "completed", ""
        if status == "interrupted":
            return (
                ("cancelled", "cancelled_by_caller")
                if active.cancelled
                else ("incomplete", "codex_turn_interrupted")
            )
        if error := _object(turn.get("error")):
            code, _ = _error_details(error)
            if code != active.error_code:
                await self._emit_error(active, error, False)
        if active.auth_required:
            return "failed", AuthRequiredError.code
        return "failed", active.error_code or "codex_turn_failed"


def _within_workspace(path: object, workspace: Path) -> bool:
    if not isinstance(path, str) or not Path(path).is_absolute():
        return False
    try:
        return Path(path).resolve().is_relative_to(workspace)
    except (OSError, RuntimeError, ValueError):
        return False


def _resolve_codex_home(profile: AgentProfile, cwd: Path, env: dict[str, str]) -> Path:
    try:
        if profile.profile_dir is not None:
            path = profile.profile_dir
        elif configured := env.get("CODEX_HOME"):
            path = cwd / configured
        else:
            if env.get("HOME") == "":
                import pwd

                home = Path(pwd.getpwuid(os.getuid()).pw_dir)
            else:
                home = Path(profile.env["HOME"]) if "HOME" in profile.env else Path.home()
            path = cwd / home / ".codex"
        state_dir = path.resolve()
    except (OSError, RuntimeError, ValueError, ImportError, KeyError):
        raise WorkspaceViolationError("Could not resolve the Codex state directory") from None
    if state_dir.is_relative_to(cwd) or cwd.is_relative_to(state_dir):
        raise WorkspaceViolationError("Codex state directory must be separate from the workspace")
    return state_dir


def _retrieve_exception(future: asyncio.Future[Any]) -> None:
    if not future.cancelled():
        future.exception()


async def _decide_approval(active: _Turn, request: ApprovalRequest) -> str:
    async def ask() -> ApprovalDecision:
        return await active.on_approval(request)

    stop = asyncio.create_task(active.approvals_closed.wait())
    answer = asyncio.create_task(ask())
    answer.add_done_callback(_retrieve_exception)
    try:
        await asyncio.wait((stop, answer), return_when=asyncio.FIRST_COMPLETED)
        if active.approvals_closed.is_set():
            return "cancel"
        if not answer.cancelled():
            decision = answer.result()
            if isinstance(decision, ApprovalDecision):
                if decision.outcome == "allow_once":
                    return "accept"
                if decision.outcome in ("deny_cancel", "cancelled"):
                    return "cancel"
    except Exception:
        return "decline"
    finally:
        stop.cancel()
        if not answer.done():
            answer.cancel()
    return "decline"


class CodexAppServerDriver(Driver):
    """Open a constrained Codex thread on a dedicated stdio connection."""

    @asynccontextmanager
    async def open(
        self,
        profile: AgentProfile,
        *,
        workspace_path: Path | None,
        access: WorkspaceAccess,
        resume_ref: str | None = None,
    ) -> AsyncIterator[CodexAppServerSession]:
        if access not in _SANDBOX_MODE:
            raise UnsupportedError("Unsupported Codex workspace access")
        policy = profile.options.get("approval_policy", "on-request")
        if policy not in ("on-request", "never"):
            raise UnsupportedError("Codex approval_policy must be on-request or never")
        ephemeral = profile.options.get("ephemeral", not profile.capabilities.resume)
        if type(ephemeral) is not bool:
            raise UnsupportedError("Codex ephemeral must be a boolean")
        if resume_ref is not None and (not profile.capabilities.resume or ephemeral):
            raise ResumeUnavailableError("Codex native resume requires a persistent profile")

        cwd = (workspace_path or Path.cwd()).resolve()
        env = os.environ | profile.env
        state_dir = _resolve_codex_home(profile, cwd, env)
        if profile.profile_dir is not None or env.get("CODEX_HOME"):
            env["CODEX_HOME"] = str(state_dir)
        session = CodexAppServerSession(profile, workspace_path, access, policy, ephemeral)
        session._peer = await JsonRpcPeer.start(
            (profile.command or "codex", *(profile.args or ("app-server",))),
            cwd=cwd,
            env=env,
            include_jsonrpc=False,
            on_notification=session._on_notification,
            on_request=session._on_request,
            on_unknown_frame=session._on_unknown_frame,
            on_eof=session._on_eof,
        )
        try:
            await session._initialize(cwd, resume_ref, bool(profile.options.get("preflight_auth")))
            yield session
        finally:
            await session._peer.close()
