"""Drive ACP v1 sessions over John's JSON-RPC transport."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from john.drivers._common import (
    _check_scope,
    _ignored_options,
    _init_timeout,
    _integer,
    _number,
    _object,
    _safe_payload,
    _string,
    _Turn,
    _turn_prompt,
)
from john.transport import (
    Frame,
    FrameQueue,
    JsonRpcError,
    JsonRpcPeer,
    MethodNotFound,
    NotificationHandler,
    RpcParams,
    StdioTransport,
    redact_text,
)
from john.types import (
    AgentProfile,
    ApprovalOutcome,
    ApprovalPolicy,
    ApprovalRequest,
    AuthRequiredError,
    BusyError,
    Capabilities,
    EventSink,
    FinishedEvent,
    PlanEvent,
    ResumeUnavailableError,
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
)

_STOP_REASONS: dict[str, TurnStatus] = {
    "end_turn": "completed",
    "cancelled": "cancelled",
    "max_tokens": "incomplete",
    "max_turn_requests": "incomplete",
    "refusal": "incomplete",
}
_REJECTED_PERMISSION = {"outcome": {"outcome": "cancelled"}}


class _OrderedAcpPeer(JsonRpcPeer):
    """Enqueue ACP updates at ingress so a prompt result cannot overtake them."""

    def __init__(
        self,
        transport: StdioTransport,
        *,
        on_notification: NotificationHandler | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(transport, **kwargs)
        self._ingress_notification = on_notification

    async def handle_frame(self, frame: Frame) -> None:
        method, params = frame.get("method"), frame.get("params")
        if (
            self._ingress_notification is not None
            and isinstance(method, str)
            and frame.get("jsonrpc", "2.0") == "2.0"
            and {"id", "result", "error"}.isdisjoint(frame)
            and (params is None or isinstance(params, (dict, list)))
        ):
            await self._ingress_notification(method, params)
        else:
            await super().handle_frame(frame)


@dataclass(slots=True)
class _AcpTurn(_Turn):
    updates: FrameQueue[tuple[str, Frame]] = field(kw_only=True)
    chunks: list[str] = field(default_factory=list)
    tools: dict[str, tuple[str, str]] = field(default_factory=dict)
    usage: Frame = field(default_factory=dict)


def _applicable_option(options: list[Frame], outcome: ApprovalOutcome) -> Frame:
    kinds = ("allow_once", "reject_once") if outcome == "allow_once" else ("reject_once",)
    if outcome != "cancelled":
        for kind in kinds:
            for option in options:
                if option.get("kind") == kind and _string(option.get("optionId")):
                    return {"outcome": {"outcome": "selected", "optionId": option["optionId"]}}
    return _REJECTED_PERMISSION


class AcpSession:
    def __init__(
        self, profile: AgentProfile, workspace: Path | None, access: WorkspaceAccess
    ) -> None:
        self._workspace = workspace
        self._access = access
        self._include_raw = profile.options.get("include_raw_other") is True
        self._ignored = _ignored_options(
            profile, {"init_timeout_s", "include_raw_other", "session_cwd"}
        )
        self._caps = Capabilities(
            approvals="live",
            thinking="stream",
            enforced_sandbox=profile.capabilities.enforced_sandbox,
        )
        self._session_id: str | None = None
        self._active: _AcpTurn | None = None
        self._broken = False
        self._peer: JsonRpcPeer

    @property
    def native_ref(self) -> str | None:
        return self._session_id

    @property
    def capabilities(self) -> Capabilities:
        return self._caps

    @property
    def ignored_fields(self) -> dict[str, str]:
        return dict(self._ignored)

    async def _initialize(self, cwd: str, resume_ref: str | None, timeout: float) -> None:
        params = {"cwd": cwd, "mcpServers": []}
        try:
            reply = _object(
                await self._peer.request(
                    "initialize",
                    {
                        "protocolVersion": 1,
                        "clientCapabilities": {"fs": {}, "auth": {}},
                        "clientInfo": {"name": "john", "version": "0.1.0"},
                    },
                    timeout_s=timeout,
                )
            )
            if _integer(reply.get("protocolVersion")) != 1:
                raise UnsupportedError("The peer did not negotiate ACP v1")
            caps = _object(reply.get("agentCapabilities"))
            sessions = _object(caps.get("sessionCapabilities", caps.get("session_capabilities")))
            load = caps.get("loadSession", caps.get("load_session"))
            supports_load = load is True or (
                load is None and isinstance(sessions.get("load"), dict)
            )
            supports_resume = isinstance(sessions.get("resume"), dict)
            self._caps = self._caps.model_copy(update={"resume": supports_load or supports_resume})
            if resume_ref is None:
                opened = _object(await self._peer.request("session/new", params, timeout_s=timeout))
                self._session_id = _string(opened.get("sessionId"))
                if not self._session_id:
                    raise TransportError("ACP session/new omitted the session ID")
                return
            if not resume_ref or not self._caps.resume:
                raise ResumeUnavailableError("The ACP peer did not advertise session restoration")
            method = "session/load" if supports_load else "session/resume"
            restored = await self._peer.request(
                method, {**params, "sessionId": resume_ref}, timeout_s=timeout
            )
            if restored is not None and (
                not isinstance(restored, dict)
                or restored.get("sessionId", resume_ref) != resume_ref
            ):
                raise ResumeUnavailableError("The ACP peer returned an invalid restored session")
            self._session_id = resume_ref
        except JsonRpcError as exc:
            if exc.code == -32000:
                raise AuthRequiredError("The ACP peer requires authentication") from None
            if resume_ref is not None and self._caps.resume:
                raise ResumeUnavailableError(
                    f"ACP session restoration failed ({exc.code})"
                ) from None
            raise TransportError(f"ACP handshake failed ({exc.code})") from None
        except TimeoutError:
            raise TransportError("ACP handshake timed out") from None

    async def _notification(self, method: str, params: RpcParams) -> None:
        if (turn := self._active) is not None and turn.dispatched:
            body = _object(params)
            if body.get("sessionId", self._session_id) == self._session_id:
                payload = body if params is None or isinstance(params, dict) else {"params": params}
                size = self._peer.transport.frame_size({"method": method, "params": params})
                turn.updates.put_nowait((method, payload), size)

    async def _unknown_frame(self, frame: Frame) -> None:
        if (turn := self._active) is not None and turn.dispatched:
            turn.updates.put_nowait(
                (_string(frame.get("method"), "jsonrpc/unknown"), frame),
                self._peer.transport.frame_size(frame),
            )

    async def _permission(self, method: str, params: RpcParams, req_id: int | str) -> Frame:
        if method != "session/request_permission":
            raise MethodNotFound
        turn, body = self._active, _object(params)
        if turn is None or not turn.dispatched or body.get("sessionId") != self._session_id:
            return _REJECTED_PERMISSION
        tool = _object(body.get("toolCall"))
        offered = body.get("options")
        options = [_object(item) for item in offered] if isinstance(offered, list) else []
        name = _string(tool.get("name")) or _string(tool.get("title"), method)
        decision = await turn.decide(
            ApprovalRequest(
                request_id=json.dumps(req_id),
                run_id=turn.run_id,
                action=name,
                summary=redact_text(_string(tool.get("title"), name)),
                args=_object(tool.get("rawInput")),
                offered_options=tuple(
                    option["kind"] for option in options if _string(option.get("kind"))
                ),
            )
        )
        if decision.outcome == "deny_cancel":
            turn.request_cancel(self._interrupt)
        return _applicable_option(options, decision.outcome)

    async def _interrupt(self) -> None:
        await self._peer.notify("session/cancel", {"sessionId": self._session_id})

    async def cancel(self, run_id: str) -> None:
        if (turn := self._active) is not None and turn.run_id == run_id:
            turn.request_cancel(self._interrupt)
            await turn.done.wait()

    async def _prompt(self, turn: _AcpTurn, text: str) -> Frame:
        try:
            if turn.cancel_requested:
                return {"stopReason": "cancelled"}
            turn.dispatched = True
            response = await self._peer.request(
                "session/prompt",
                {
                    "sessionId": self._session_id,
                    "prompt": [{"type": "text", "text": text}],
                },
            )
            return _object(response)
        finally:
            turn.stopped.set()
            turn.updates.finish()

    async def run_turn(
        self, request: TurnRequest, on_event: EventSink, on_approval: ApprovalPolicy
    ) -> TurnFinished:
        if self._active is not None:
            raise BusyError("The ACP session already has an active turn")
        transport = self._peer.transport
        transport.raise_if_failed()
        if self._broken or transport.closed:
            raise TransportError("The ACP session is closed")
        prompt = _turn_prompt(request, self._workspace, self._access)
        turn = self._active = _AcpTurn(
            request, on_event, on_approval, self._include_raw, updates=FrameQueue(transport)
        )
        try:
            await turn.emit(
                StartedEvent, native_session_ref=self.native_ref, ignored_fields=self.ignored_fields
            )
            native = asyncio.create_task(self._prompt(turn, prompt))
            try:
                while (entry := await turn.updates.get()) is not None:
                    transport.raise_if_failed()
                    update, _ = entry
                    await self._emit_update(turn, *update)
                transport.raise_if_failed()
                reply = await native
                stop = _string(reply.get("stopReason"))
                if stop not in _STOP_REASONS:
                    raise TransportError("ACP session/prompt returned an unknown stop reason")
                if usage := _object(reply.get("usage")):
                    await self._usage(turn, usage)
                transport.raise_if_failed()
                status, reason = _STOP_REASONS[stop], stop if stop != "end_turn" else ""
            except JsonRpcError as exc:
                status = "failed"
                reason = "auth_required" if exc.code == -32000 else f"prompt_rpc_error: {exc.code}"
            except TransportError as exc:
                self._broken = True
                status, reason = "indeterminate", redact_text(str(exc))
            finally:
                if not native.done():
                    self._broken |= turn.dispatched
                    native.cancel()
                await asyncio.gather(native, return_exceptions=True)
            await turn.emit(FinishedEvent, status=status, reason=reason)
            return TurnFinished(
                result=TurnResult(
                    run_id=turn.run_id,
                    status=status,
                    reason=reason,
                    text=redact_text("".join(turn.chunks)),
                    usage=turn.usage or None,
                    native_session_ref=self.native_ref,
                )
            )
        finally:
            turn.updates.clear()
            try:
                await turn.finish()
            finally:
                self._active = None

    async def _emit_update(self, turn: _AcpTurn, method: str, params: Frame) -> None:
        if method != "session/update":
            await turn.other(method, params)
            return
        update = _object(params.get("update"))
        kind = _string(update.get("sessionUpdate"), method)
        match kind:
            case "agent_message_chunk" | "agent_thought_chunk":
                content = _object(update.get("content"))
                if content.get("type") != "text" or not isinstance(content.get("text"), str):
                    await turn.other(kind, params)
                    return
                text = content["text"]
                if kind == "agent_message_chunk":
                    turn.chunks.append(text)
                    await turn.emit(TextEvent, text=redact_text(text), is_delta=True)
                else:
                    await turn.emit(ThoughtEvent, text=redact_text(text), is_delta=True)
            case "tool_call" | "tool_call_update" if _string(update.get("toolCallId")):
                await self._tool(turn, kind, update)
            case "plan" | "plan_update":
                plan = _object(update.get("plan")) or update
                if isinstance(entries := plan.get("entries"), list):
                    items = tuple(
                        redact_text(entry["content"])
                        for entry in entries
                        if _string(_object(entry).get("content"))
                    )
                    await turn.emit(PlanEvent, items=items)
                elif isinstance(content := plan.get("content"), str):
                    await turn.emit(PlanEvent, items=(redact_text(content),))
                else:
                    await turn.other(kind, params)
            case "usage_update":
                await self._usage(
                    turn,
                    {
                        key: value
                        for key, value in update.items()
                        if key not in ("sessionUpdate", "_meta")
                    },
                )
            case _:
                await turn.other(kind, params)

    async def _tool(self, turn: _AcpTurn, kind: str, update: Frame) -> None:
        call_id = update["toolCallId"]
        previous_name, previous_title = turn.tools.get(call_id, ("", ""))
        title = _string(update.get("title")) or previous_title
        name = (
            _string(update.get("name"))
            or previous_name
            or title
            or _string(update.get("kind"), "tool")
        )
        turn.tools[call_id] = (name, title)
        state = update.get("status")
        phase: Literal["started", "updated", "completed", "failed"]
        if state == "completed":
            phase = "completed"
        elif state in ("failed", "cancelled"):
            phase = "failed"
        else:
            phase = "started" if kind == "tool_call" else "updated"
        await turn.emit(
            ToolEvent,
            call_id=redact_text(call_id),
            name=redact_text(name),
            phase=phase,
            summary=redact_text(title),
        )

    async def _usage(self, turn: _AcpTurn, usage: Frame) -> None:
        turn.usage.update(_safe_payload(usage))
        cost = _object(usage.get("cost"))
        await turn.emit(
            UsageEvent,
            input_tokens=_integer(usage.get("inputTokens")),
            output_tokens=_integer(usage.get("outputTokens")),
            cost_usd=_number(cost.get("amount")) if cost.get("currency") == "USD" else None,
        )

    async def aclose(self) -> None:
        await self._peer.close()


class AcpDriver:
    @asynccontextmanager
    async def open(
        self,
        profile: AgentProfile,
        *,
        workspace_path: Path | None,
        access: WorkspaceAccess,
        resume_ref: str | None = None,
    ) -> AsyncIterator[AcpSession]:
        if not profile.command or not profile.command.strip():
            raise TransportError("ACP requires an explicit command")
        _check_scope(profile, workspace_path, access)
        cwd = profile.options.get("session_cwd", str((workspace_path or Path.cwd()).resolve()))
        if (
            not isinstance(cwd, str)
            or not PurePosixPath(cwd).is_absolute()
            or ".." in cwd.split("/")
            or "\x00" in cwd
        ):
            raise UnsupportedError("session_cwd must be an absolute agent-side path")
        timeout = _init_timeout(profile)
        session = AcpSession(profile, workspace_path, access)
        session._peer = await _OrderedAcpPeer.start(
            (profile.command, *profile.args),
            cwd=workspace_path,
            env={**os.environ, **profile.env},
            include_jsonrpc=True,
            on_notification=session._notification,
            on_request=session._permission,
            on_unknown_frame=session._unknown_frame,
        )
        try:
            await session._initialize(cwd, resume_ref, timeout)
            yield session
        finally:
            await session.aclose()
