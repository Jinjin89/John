"""Drive Claude Code's bidirectional stream-json interface over stdio."""

from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from itertools import count
from pathlib import Path

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
    build_child_env,
)
from john.transport import Frame, FrameQueue, StdioTransport, redact_text
from john.types import (
    AgentProfile,
    ApprovalDecision,
    ApprovalPolicy,
    ApprovalRequest,
    AuthRequiredError,
    BusyError,
    Capabilities,
    ErrorEvent,
    EventSink,
    FinishedEvent,
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

_INIT_ID = "john-init-1"
_STDIO_FLAGS = (
    "-p",
    "--input-format",
    "stream-json",
    "--output-format",
    "stream-json",
    "--verbose",
    "--include-partial-messages",
    "--permission-prompts",
    "host",
    "--permission-prompt-tool",
    "stdio",
)
_CANCELLED = {"aborted_streaming", "aborted_tools", "cancelled", "canceled", "interrupted"}
_AUTH_ERROR_TYPES = {"authentication_error", "authentication_failed", "invalid_api_key"}
_UNAUTHORIZED_TEXT = re.compile(r"\b(?:401|unauthorized)\b")
_FORBIDDEN_TEXT = re.compile(r"\b(?:403|permission_error)\b")


def _auth_required(message: str, *, permission_denied: bool = False) -> bool:
    message = message.casefold().replace("’", "'")
    return any(
        phrase in message
        for phrase in (
            "authentication required",
            "authentication_error",
            "authentication failed",
            "invalid api key",
            "invalid_api_key",
            "not logged in",
            "please run /login",
            "oauth token has expired",
            "invalid authentication",
            "your account can't receive replies",
            "verification portal",
        )
    ) or (
        not permission_denied
        and _FORBIDDEN_TEXT.search(message) is None
        and _UNAUTHORIZED_TEXT.search(message) is not None
    )


def _error_details(frame: Frame, message: str = "") -> tuple[str, str]:
    raw_error = frame.get("error")
    error = _object(raw_error)
    error_type = _string(error.get("type")) or _string(raw_error)
    error_code = _string(error.get("code"))
    error_message = _string(error.get("message")) or _string(raw_error)
    status = frame.get("api_error_status")
    permission_denied = status == 403 or "permission_error" in (error_type, error_code)
    auth = (
        status == 401
        or error_type in _AUTH_ERROR_TYPES
        or error_code in _AUTH_ERROR_TYPES
        or _auth_required(message, permission_denied=permission_denied)
        or _auth_required(error_message, permission_denied=permission_denied)
    )
    code = AuthRequiredError.code if auth else "api_error"
    return code, redact_text(message or error_message or error_type or error_code)


@dataclass(slots=True)
class _Block:
    kind: str
    deltas: list[str] = field(default_factory=list)
    snapshot: str | None = None

    @property
    def text(self) -> str:
        return self.snapshot if self.snapshot is not None else "".join(self.deltas)


@dataclass(slots=True)
class _Message:
    message_id: str = ""
    blocks: dict[int, _Block] = field(default_factory=dict)
    focus: int | None = None

    def snapshot_index(self, kind: str) -> int:
        # The CLI may send only the just-completed block, even for a multi-block message.
        if self.focus is not None and self.blocks[self.focus].kind == kind:
            return self.focus
        return next(
            (index for index, block in self.blocks.items() if block.kind == kind),
            max(self.blocks, default=-1) + 1,
        )


@dataclass(slots=True)
class _Tool:
    name: str
    finished: bool = False


@dataclass(slots=True)
class _ClaudeTurn(_Turn):
    inbox: FrameQueue[Frame] = field(kw_only=True)
    live: dict[str, _Message] = field(default_factory=dict)
    by_id: dict[tuple[str, str], _Message] = field(default_factory=dict)
    top_level: list[_Message] = field(default_factory=list)
    tools: dict[str, _Tool] = field(default_factory=dict)
    seen_snapshots: set[str] = field(default_factory=set)
    started: bool = False
    api_error: tuple[str, str] | None = None

    def new_message(self, parent: str, message_id: str = "") -> _Message:
        message = self.live[parent] = _Message(message_id)
        if not parent:
            self.top_level.append(message)
        if message_id:
            self.by_id[parent, message_id] = message
        return message

    def snapshot_message(self, parent: str, message_id: str) -> _Message:
        if message_id and (existing := self.by_id.get((parent, message_id))) is not None:
            return existing
        active = self.live.get(parent)
        if active is None or (message_id and active.message_id and active.message_id != message_id):
            return self.new_message(parent, message_id)
        if message_id:
            self.by_id[parent, message_id] = active
            active.message_id = message_id
        return active

    def text(self) -> str:
        return "".join(
            block.text
            for message in self.top_level
            for _, block in sorted(message.blocks.items())
            if block.kind == "text"
        )


class ClaudeStreamJsonSession:
    def __init__(
        self,
        profile: AgentProfile,
        workspace: Path | None,
        access: WorkspaceAccess,
        resume_ref: str | None,
        mode: str,
    ) -> None:
        self._workspace = workspace
        self._access = access
        self._session_id = resume_ref
        self._include_raw = profile.options.get("include_raw_other") is True
        self._ignored = _ignored_options(
            profile, {"init_timeout_s", "include_raw_other", "permission_mode"}
        )
        self._caps = Capabilities(
            approvals="live"
            if mode in {"manual", "default", "plan", "acceptEdits"}
            else "auto_only",
            resume=True,
            thinking="stream",
            enforced_sandbox=profile.capabilities.enforced_sandbox,
        )
        self._initialized: asyncio.Future[Frame | None] = asyncio.get_running_loop().create_future()
        self._transport: StdioTransport
        self._controls: dict[str, asyncio.Task[None]] = {}
        self._interrupt_ids = count(1)
        self._active: _ClaudeTurn | None = None
        self._broken = False

    @property
    def native_ref(self) -> str | None:
        return self._session_id

    @property
    def capabilities(self) -> Capabilities:
        return self._caps

    @property
    def ignored_fields(self) -> dict[str, str]:
        return dict(self._ignored)

    async def _initialize(self, timeout: float, resume_ref: str | None, mode: str) -> None:
        try:
            async with asyncio.timeout(timeout):
                await self._send_control(_INIT_ID, "initialize")
                response = await self._initialized
        except TimeoutError:
            raise TransportError("Claude initialize timed out") from None
        self._transport.raise_if_failed()
        if response is not None and response.get("subtype") == "success":
            effective = _string(_object(response.get("response")).get("current_permission_mode"))
            if effective and effective != mode and {effective, mode} != {"manual", "default"}:
                raise UnsupportedError("Claude initialized with a different permission mode")
            return
        if response is not None and _error_details(response)[0] == AuthRequiredError.code:
            raise AuthRequiredError("Claude requires authentication")
        if resume_ref is not None:
            raise ResumeUnavailableError("Claude exited or rejected initialization during resume")
        raise TransportError("Claude exited or rejected stream initialization")

    async def _send_control(self, request_id: str, subtype: str) -> None:
        await self._transport.send_json(
            {
                "type": "control_request",
                "request_id": request_id,
                "request": {"subtype": subtype},
            }
        )

    async def _on_frame(self, frame: Frame) -> None:
        turn = self._active
        match frame.get("type"):
            case "control_response":
                response = _object(frame.get("response"))
                if response.get("request_id") == _INIT_ID and not self._initialized.done():
                    self._initialized.set_result(response)
                return
            case "control_request":
                request_id = _string(frame.get("request_id"))
                if not request_id:
                    raise TransportError("Claude control request omitted its request ID")
                if request_id not in self._controls:
                    task = asyncio.create_task(
                        self._control(turn, request_id, _object(frame.get("request")))
                    )
                    self._controls[request_id] = task
                    task.add_done_callback(lambda done: self._control_done(request_id, done))
                return
            case "control_cancel_request":
                if task := self._controls.pop(_string(frame.get("request_id")), None):
                    task.cancel()
                return
        if turn is not None and turn.dispatched:
            session_id = _string(frame.get("session_id"))
            if session_id and frame.get("type") == "system" and frame.get("subtype") == "init":
                self._session_id = session_id
            if session_id and self._session_id and session_id != self._session_id:
                return
            if frame.get("type") == "result":
                turn.stopped.set()
            turn.inbox.put_nowait(frame, self._transport.frame_size(frame))

    async def _on_eof(self, returncode: int | None) -> None:
        if not self._initialized.done():
            self._initialized.set_result(None)
        if (turn := self._active) is not None:
            turn.stopped.set()
            turn.inbox.finish()

    def _control_done(self, request_id: str, task: asyncio.Task[None]) -> None:
        if self._controls.get(request_id) is task:
            del self._controls[request_id]
        if not task.cancelled() and task.exception() is not None:
            self._broken = True
            if (turn := self._active) is not None:
                turn.stopped.set()
                turn.inbox.finish()

    async def _control(self, turn: _ClaudeTurn | None, request_id: str, request: Frame) -> None:
        response: Frame = {"request_id": request_id}
        decision = ApprovalDecision()
        if request.get("subtype") == "can_use_tool":
            inputs = _object(request.get("input"))
            tool_name = _string(request.get("tool_name"))
            if (
                turn is not None
                and turn.dispatched
                and isinstance(request.get("input"), dict)
                and tool_name
            ):
                decision = await turn.decide(
                    ApprovalRequest(
                        request_id=request_id,
                        run_id=turn.run_id,
                        action=tool_name,
                        summary=redact_text(_string(request.get("description"), tool_name)),
                        args=deepcopy(inputs),
                        offered_options=("allow_once", "deny"),
                    )
                )
            if decision.outcome == "allow_once":
                answer = {"behavior": "allow", "updatedInput": inputs}
            else:
                answer = {
                    "behavior": "deny",
                    "message": redact_text(decision.reason) or "Denied by John policy",
                }
            response.update(subtype="success", response=answer)
        else:
            response.update(subtype="error", error="Unsupported control request")
        await self._transport.send_json({"type": "control_response", "response": response})
        if decision.outcome == "deny_cancel" and turn is not None:
            self._request_cancel(turn)

    def _request_cancel(self, turn: _ClaudeTurn) -> None:
        async def interrupt() -> None:
            await self._send_control(f"john-interrupt-{next(self._interrupt_ids)}", "interrupt")

        turn.request_cancel(interrupt)

    async def cancel(self, run_id: str) -> None:
        if (turn := self._active) is not None and turn.run_id == run_id:
            self._request_cancel(turn)
            await turn.done.wait()

    async def _started(self, turn: _ClaudeTurn) -> None:
        if not turn.started:
            turn.started = True
            await turn.emit(
                StartedEvent, native_session_ref=self.native_ref, ignored_fields=self.ignored_fields
            )

    async def run_turn(
        self, request: TurnRequest, on_event: EventSink, on_approval: ApprovalPolicy
    ) -> TurnFinished:
        if self._active is not None:
            raise BusyError("The Claude session already has an active turn")
        self._transport.raise_if_failed()
        if self._broken or self._transport.closed:
            raise TransportError("The Claude session is closed")
        prompt = _turn_prompt(request, self._workspace, self._access)
        turn = self._active = _ClaudeTurn(
            request, on_event, on_approval, self._include_raw, inbox=FrameQueue(self._transport)
        )
        terminal = False
        try:
            try:
                turn.dispatched = True
                await self._transport.send_json(
                    {
                        "type": "user",
                        "message": {"role": "user", "content": [{"type": "text", "text": prompt}]},
                        "parent_tool_use_id": None,
                        "session_id": self.native_ref or "",
                    }
                )
                result = await self._consume_turn(turn)
                self._transport.raise_if_failed()
                terminal = True
            except TransportError as exc:
                result = TurnResult(
                    run_id=turn.run_id,
                    status="indeterminate",
                    reason=redact_text(str(exc)),
                    text=redact_text(turn.text()),
                    native_session_ref=self.native_ref,
                )
            await self._started(turn)
            await turn.emit(FinishedEvent, status=result.status, reason=result.reason)
            return TurnFinished(result=result)
        finally:
            turn.inbox.clear()
            if turn.dispatched and not terminal:
                self._broken = True
            try:
                await turn.finish()
            finally:
                self._active = None

    async def _consume_turn(self, turn: _ClaudeTurn) -> TurnResult:
        while (entry := await turn.inbox.get()) is not None:
            self._transport.raise_if_failed()
            frame, _ = entry
            kind = _string(frame.get("type"), "unknown")
            subtype = _string(frame.get("subtype"))
            if kind == "system" and subtype == "init":
                await self._started(turn)
                continue
            if kind == "result":
                self._session_id = _string(frame.get("session_id")) or self._session_id
                await self._started(turn)
                return await self._result(turn, frame)
            if kind == "user":
                await self._tool_results(turn, frame)
                continue
            if kind == "system" and subtype == "status":
                continue
            await self._started(turn)
            if kind == "stream_event":
                await self._stream(turn, frame)
            elif kind == "assistant":
                await self._assistant(turn, frame)
            else:
                await turn.other(
                    f"system/{subtype}" if kind == "system" and subtype else kind, frame
                )
        self._transport.raise_if_failed()
        raise TransportError("Claude stdout closed before a terminal result")

    async def _tool_results(self, turn: _ClaudeTurn, frame: Frame) -> None:
        content = _object(frame.get("message")).get("content")
        if not isinstance(content, list):
            return
        for item in content:
            block = _object(item)
            if block.get("type") != "tool_result":
                continue
            call_id = _string(block.get("tool_use_id"))
            tool = turn.tools.get(call_id)
            if tool is None or tool.finished:
                continue
            tool.finished = True
            output = block.get("content")
            if isinstance(output, list):
                output = " ".join(
                    _string(part.get("text"))
                    for part in output
                    if isinstance(part, dict) and part.get("type") == "text"
                )
            await turn.emit(
                ToolEvent,
                call_id=redact_text(call_id),
                name=tool.name,
                phase="failed" if block.get("is_error") is True else "completed",
                summary=redact_text(_string(output).strip())[:512],
            )

    async def _stream(self, turn: _ClaudeTurn, frame: Frame) -> None:
        parent = _string(frame.get("parent_tool_use_id"))
        event = _object(frame.get("event"))
        kind = _string(event.get("type"))
        if kind == "message_start":
            turn.new_message(parent, _string(_object(event.get("message")).get("id")))
            return
        if kind in {"message_stop", "message_delta", "content_block_stop", "ping"}:
            return
        index = _integer(event.get("index"))
        if index is None or kind not in {"content_block_start", "content_block_delta"}:
            await turn.other(f"stream_event/{kind or 'unknown'}", frame)
            return
        message = turn.live.get(parent) or turn.new_message(parent)
        if kind == "content_block_start":
            await self._content(
                turn, message, index, _object(event.get("content_block")), snapshot=False
            )
            return
        delta = _object(event.get("delta"))
        match delta.get("type"):
            case "text_delta":
                await self._content(
                    turn,
                    message,
                    index,
                    {"type": "text", "text": delta.get("text")},
                    snapshot=False,
                )
            case "thinking_delta":
                await self._content(
                    turn,
                    message,
                    index,
                    {"type": "thinking", "thinking": delta.get("thinking")},
                    snapshot=False,
                )
            case "input_json_delta" | "signature_delta":
                pass
            case _:
                await turn.other("stream_event/content_block_delta", frame)

    async def _assistant(self, turn: _ClaudeTurn, frame: Frame) -> None:
        message = _object(frame.get("message"))
        content = message.get("content")
        if (
            frame.get("is_api_error_message") is True
            or frame.get("error")
            or frame.get("api_error_status") is not None
        ):
            detail = (
                " ".join(
                    _string(item.get("text"))
                    for item in content
                    if isinstance(item, dict) and item.get("type") == "text"
                ).strip()
                if isinstance(content, list)
                else ""
            )
            code, detail = _error_details(frame, detail)
            detail = detail or "Claude API error"
            turn.api_error = (code, detail)
            await turn.emit(ErrorEvent, code=code, message=detail)
            return
        if not isinstance(content, list):
            await turn.other("assistant", frame)
            return
        snapshot_id = _string(frame.get("uuid"))
        if snapshot_id:
            if snapshot_id in turn.seen_snapshots:
                return
            turn.seen_snapshots.add(snapshot_id)
        model_message = turn.snapshot_message(
            _string(frame.get("parent_tool_use_id")), _string(message.get("id"))
        )
        for index, body in enumerate(content):
            block = _object(body)
            if len(content) == 1:
                index = model_message.snapshot_index(_string(block.get("type")))
            await self._content(turn, model_message, index, block, snapshot=True)

    async def _content(
        self, turn: _ClaudeTurn, message: _Message, index: int, content: Frame, *, snapshot: bool
    ) -> None:
        kind = _string(content.get("type"), "unknown")
        block = message.blocks.setdefault(index, _Block(kind))
        if not snapshot:
            message.focus = index
        if kind == "tool_use":
            call_id = _string(content.get("id"))
            if call_id and call_id not in turn.tools:
                name = redact_text(_string(content.get("name"), "tool"))
                turn.tools[call_id] = _Tool(name)
                await turn.emit(
                    ToolEvent,
                    call_id=redact_text(call_id),
                    name=name,
                    phase="started",
                )
            return
        if kind == "redacted_thinking":
            return
        if kind not in {"text", "thinking"} or block.kind != kind:
            await turn.other(f"content/{kind}", content)
            return
        value = _string(content.get(kind))
        event = TextEvent if kind == "text" else ThoughtEvent
        if snapshot:
            previous = block.text
            block.snapshot = value
            if value.startswith(previous):
                if suffix := value[len(previous) :]:
                    await turn.emit(event, text=redact_text(suffix), is_delta=bool(previous))
            elif kind == "text":
                await turn.emit(TextEvent, text=redact_text(value), is_delta=False)
        elif value and block.snapshot is None:
            block.deltas.append(value)
            await turn.emit(event, text=redact_text(value), is_delta=True)

    async def _result(self, turn: _ClaudeTurn, frame: Frame) -> TurnResult:
        subtype = _string(frame.get("subtype"))
        terminal = _string(frame.get("terminal_reason"))
        status: TurnStatus
        if {subtype, terminal, _string(frame.get("stop_reason"))} & _CANCELLED:
            status, reason = "cancelled", "cancelled"
        elif subtype == "success" and frame.get("is_error") is False:
            status, reason = "completed", ""
        elif frame.get("is_error") is True or subtype.startswith("error_") or turn.api_error:
            status = "failed"
            errors = frame.get("errors")
            details = (
                " ".join(item for item in errors if isinstance(item, str))
                if isinstance(errors, list)
                else ""
            )
            code, detail = _error_details(frame, details or _string(frame.get("result")))
            if turn.api_error and not detail and code == "api_error":
                earlier_code, earlier_detail = turn.api_error
                http_status = frame.get("api_error_status")
                if http_status is None:
                    code, detail = earlier_code, earlier_detail
                elif http_status == 403 and _auth_required(earlier_detail, permission_denied=True):
                    code = AuthRequiredError.code
            reason = (
                code
                if code == AuthRequiredError.code
                else detail
                or redact_text(terminal or (subtype if subtype != "success" else "api_error"))
            )
        else:
            raise TransportError("Claude returned an unrecognized terminal result")
        usage = _safe_payload(_object(frame.get("usage")))
        cost = _number(frame.get("total_cost_usd"))
        if cost is not None:
            usage["total_cost_usd"] = cost
        if usage:
            await turn.emit(
                UsageEvent,
                input_tokens=_integer(usage.get("input_tokens")),
                output_tokens=_integer(usage.get("output_tokens")),
                cost_usd=cost,
            )
        text = _string(frame.get("result"), turn.text()) if status == "completed" else turn.text()
        return TurnResult(
            run_id=turn.run_id,
            status=status,
            reason=reason,
            text=redact_text(text),
            structured=frame.get("structured_output"),
            has_structured="structured_output" in frame,
            usage=usage or None,
            native_session_ref=self.native_ref,
        )

    async def aclose(self) -> None:
        try:
            await self._transport.close()
        finally:
            tasks = tuple(self._controls.values())
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)


class ClaudeStreamJsonDriver:
    @asynccontextmanager
    async def open(
        self,
        profile: AgentProfile,
        *,
        workspace_path: Path | None,
        access: WorkspaceAccess,
        resume_ref: str | None = None,
    ) -> AsyncIterator[ClaudeStreamJsonSession]:
        _check_scope(profile, workspace_path, access)
        timeout = _init_timeout(profile)
        mode = profile.options.get("permission_mode", "manual")
        if not isinstance(mode, str) or not mode.strip():
            raise UnsupportedError("permission_mode must be a nonempty Claude mode")
        if resume_ref is not None and (not resume_ref or resume_ref.startswith("-")):
            raise ResumeUnavailableError("Claude resume requires a valid session ID")
        resume = ("--resume", resume_ref) if resume_ref is not None else ()
        session = ClaudeStreamJsonSession(profile, workspace_path, access, resume_ref, mode)
        session._transport = await StdioTransport.start(
            (
                profile.command or "claude",
                *_STDIO_FLAGS,
                "--permission-mode",
                mode,
                *resume,
                *profile.args,
            ),
            cwd=workspace_path,
            env=build_child_env(profile),
            on_frame=session._on_frame,
            on_eof=session._on_eof,
        )
        try:
            await session._initialize(timeout, resume_ref, mode)
            yield session
        finally:
            await session.aclose()
