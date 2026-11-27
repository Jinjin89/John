"""Run trusted asynchronous specialists in the John process."""

from __future__ import annotations

import asyncio
import csv
import hashlib
import io
import json
import os
import stat
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Mapping
from contextlib import asynccontextmanager, contextmanager, suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, BinaryIO, Literal
from uuid import uuid4
from warnings import warn

from pydantic import AliasChoices, BaseModel, Field

from john.types import (
    AgentProfile,
    ApprovalDecision,
    ApprovalEvent,
    ApprovalPolicy,
    ApprovalRequest,
    BusyError,
    Capabilities,
    Driver,
    DriverSession,
    ErrorEvent,
    EventSink,
    FinishedEvent,
    InvalidOutputError,
    JohnError,
    ResumeUnavailableError,
    RuntimeEvent,
    StartedEvent,
    TextEvent,
    ThoughtEvent,
    ToolEvent,
    TurnFinished,
    TurnNeedsInput,
    TurnOutcome,
    TurnRequest,
    TurnResult,
    UnsupportedError,
    WorkspaceAccess,
    WorkspaceRef,
    WorkspaceViolationError,
)

type SpecialistHandler = Callable[[SpecialistContext, TurnRequest], Awaitable[TurnOutcome]]
type EventFactory = Callable[..., RuntimeEvent]

_CUTOFF_QUESTION = "Which cutoff threshold should be applied?"
_READ_ONLY: frozenset[WorkspaceAccess] = frozenset({"read_only"})
_SAFE_OPEN_SUPPORTED = os.open in os.supports_dir_fd and all(
    hasattr(os, flag) for flag in ("O_DIRECTORY", "O_NOFOLLOW", "O_NONBLOCK")
)


class SpecialistContext:
    """Per-turn callbacks and workspace binding for a trusted specialist."""

    def __init__(
        self,
        profile: AgentProfile,
        workspace_path: Path | None,
        access: WorkspaceAccess,
        resume_ref: str | None,
        run_id: str,
        on_event: EventSink,
        on_approval: ApprovalPolicy,
    ) -> None:
        self.profile = profile
        self.workspace_path = workspace_path
        self.access = access
        self.resume_ref = resume_ref
        self.run_id = run_id
        self.cancelled = asyncio.Event()
        self._on_event = on_event
        self._on_approval = on_approval
        self._next_seq = 0
        self._event_lock = asyncio.Lock()
        self._ended = False

    async def emit(self, event: RuntimeEvent | EventFactory, **fields: Any) -> None:
        """Publish a typed event or call a factory with run_id, seq, and fields."""
        await self._publish(event, **fields)

    async def _publish(
        self, event: RuntimeEvent | EventFactory, *, lifecycle: bool = False, **fields: Any
    ) -> None:
        async with self._event_lock:
            if self._ended:
                raise RuntimeError("The specialist turn has ended")
            stamp = {"run_id": self.run_id, "seq": self._next_seq}
            if callable(event):
                event = event(**stamp, **fields)
            elif fields:
                raise TypeError("Event fields require an event factory")
            if not lifecycle and isinstance(event, StartedEvent | FinishedEvent):
                raise InvalidOutputError("The session owns the started and finished events")
            self._next_seq += 1
            if isinstance(event, FinishedEvent):
                self._ended = True
            await self._on_event(event.model_copy(update=stamp))

    async def request_approval(
        self, action: str, summary: str, args: dict[str, Any] | None = None
    ) -> ApprovalDecision:
        """Ask the host policy and record a correlated decision."""
        cancelled = ApprovalDecision(outcome="cancelled", reason="Turn cancelled")
        if self.cancelled.is_set():
            return cancelled
        request = ApprovalRequest(
            request_id=uuid4().hex,
            run_id=self.run_id,
            action=action,
            summary=summary,
            args=args or {},
            offered_options=("allow_once", "deny", "deny_cancel"),
        )
        await self.emit(
            ApprovalEvent, request_id=request.request_id, action=action, phase="requested"
        )
        decision = cancelled
        try:
            if not self.cancelled.is_set():
                decision = await self._policy_decision(request)
        except Exception:
            decision = ApprovalDecision(outcome="deny", reason="Approval policy failed")
        finally:
            if self.cancelled.is_set():
                decision = cancelled
            await self.emit(
                ApprovalEvent,
                request_id=request.request_id,
                action=action,
                phase="decided",
                outcome=decision.outcome,
            )
        return decision

    async def _policy_decision(self, request: ApprovalRequest) -> ApprovalDecision:
        pending = asyncio.ensure_future(self._on_approval(request))
        try:
            await asyncio.wait((pending,))
        except asyncio.CancelledError:
            pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)
            raise
        if pending.cancelled():
            return ApprovalDecision(outcome="deny", reason="Approval policy cancelled")
        decision = pending.result()
        if not isinstance(decision, ApprovalDecision):
            raise TypeError("Approval policy must return an ApprovalDecision")
        return decision


@dataclass(slots=True)
class _ActiveTurn:
    context: SpecialistContext
    cancel_deadline: asyncio.Future[float]
    worker: asyncio.Task[TurnOutcome] | None = None
    done: asyncio.Event = field(default_factory=asyncio.Event)

    def cancel(self, timeout_s: float) -> None:
        if self.cancel_deadline.done():
            return
        self.cancel_deadline.set_result(asyncio.get_running_loop().time() + timeout_s)
        self.context.cancelled.set()
        if self.worker is not None:
            self.worker.cancel()


class InProcessSession(DriverSession):
    """Bind a specialist to a workspace and run one turn at a time."""

    def __init__(
        self,
        handler: SpecialistHandler,
        profile: AgentProfile,
        workspace_path: Path | None,
        access: WorkspaceAccess,
        resume_ref: str | None,
        capabilities: Capabilities,
    ) -> None:
        self._handler = handler
        self._profile = profile
        self._workspace_path = workspace_path
        self._access = access
        self._native_ref = resume_ref
        self._capabilities = capabilities
        timeout = profile.options.get("close_timeout_s", 5.0)
        try:
            if type(timeout) not in (int, float):
                raise ValueError
            self._close_timeout_s = float(timeout)
            if not 0 <= self._close_timeout_s < float("inf"):
                raise ValueError
        except (ValueError, OverflowError):
            raise UnsupportedError("close_timeout_s must be finite and nonnegative") from None
        self._active: _ActiveTurn | None = None
        self._closed = False

    @property
    def native_ref(self) -> str | None:
        return self._native_ref

    @property
    def capabilities(self) -> Capabilities:
        return self._capabilities

    @property
    def ignored_fields(self) -> dict[str, str]:
        return {}

    async def run_turn(
        self, request: TurnRequest, on_event: EventSink, on_approval: ApprovalPolicy
    ) -> TurnOutcome:
        if self._active is not None:
            raise BusyError("The in-process session already has an active turn")
        if self._closed:
            raise UnsupportedError("The in-process session is closed")
        if request.outputs:
            raise UnsupportedError("In-process specialists require read-only turns")
        context = SpecialistContext(
            self._profile,
            self._workspace_path,
            self._access,
            self.native_ref,
            request.run_id,
            on_event,
            on_approval,
        )
        active = _ActiveTurn(context, asyncio.get_running_loop().create_future())
        self._active = active
        try:
            await context._publish(StartedEvent, lifecycle=True, native_session_ref=self.native_ref)
            outcome = await self._run_specialist(active, request)
            if isinstance(outcome, TurnFinished):
                result = outcome.result
                self._native_ref = result.native_session_ref or self.native_ref
                await context._publish(
                    FinishedEvent, lifecycle=True, status=result.status, reason=result.reason
                )
            else:
                self._native_ref = outcome.native_session_ref or self.native_ref
            return outcome
        finally:
            context._ended = True
            active.done.set()
            worker = active.worker
            if worker is not None and not worker.done():
                worker.add_done_callback(lambda finished: self._release_turn(active, finished))
            else:
                self._release_turn(active, worker)

    def _release_turn(
        self, active: _ActiveTurn, worker: asyncio.Future[TurnOutcome] | None
    ) -> None:
        if worker is not None and not worker.cancelled():
            worker.exception()
        if self._active is active:
            self._active = None

    async def _run_specialist(self, active: _ActiveTurn, request: TurnRequest) -> TurnOutcome:
        context = active.context
        if not context.cancelled.is_set():
            worker = asyncio.create_task(
                self._invoke(context, request), name=f"specialist:{request.run_id}"
            )
            active.worker = worker
            try:
                # Waiting on the task separately preserves cancellation of the run_turn caller.
                await asyncio.wait(
                    (worker, active.cancel_deadline), return_when=asyncio.FIRST_COMPLETED
                )
                if not worker.done():
                    remaining = active.cancel_deadline.result() - asyncio.get_running_loop().time()
                    await asyncio.wait((worker,), timeout=max(0, remaining))
            except asyncio.CancelledError:
                active.cancel(self._close_timeout_s)
                raise

            if not worker.done():
                return TurnFinished(
                    result=TurnResult(
                        run_id=request.run_id, status="indeterminate", reason="cancel_timeout"
                    )
                )
            try:
                outcome = worker.result()
            except asyncio.CancelledError:
                if not context.cancelled.is_set():
                    raise
            except Exception as exc:
                if not context.cancelled.is_set():
                    reason = f"Specialist failed ({type(exc).__name__})"
                    code = exc.code if isinstance(exc, JohnError) else "specialist_error"
                    if isinstance(exc, TimeoutError):
                        code = "turn_timeout"
                        reason = "Specialist turn timed out"
                    await context.emit(ErrorEvent, code=code, message=reason)
                    return TurnFinished(
                        result=TurnResult(run_id=request.run_id, status="failed", reason=reason)
                    )
            else:
                if isinstance(outcome, TurnFinished) or not context.cancelled.is_set():
                    return outcome
        return TurnFinished(
            result=TurnResult(
                run_id=request.run_id, status="cancelled", reason="cancelled_by_caller"
            )
        )

    async def _invoke(self, context: SpecialistContext, request: TurnRequest) -> TurnOutcome:
        async with asyncio.timeout(request.timeout_s) as deadline:
            outcome = await self._handler(context, request)
        if deadline.expired() and not (
            context.cancelled.is_set() and isinstance(outcome, TurnFinished)
        ):
            raise TimeoutError("Specialist turn timed out")
        if isinstance(outcome, TurnFinished) and outcome.result.run_id == request.run_id:
            return outcome
        if isinstance(outcome, TurnNeedsInput) and outcome.run_id == request.run_id:
            if not self.capabilities.task_questions:
                raise UnsupportedError("The specialist must declare support for task questions")
            try:
                json.dumps(outcome.continuation, allow_nan=False)
            except (TypeError, ValueError):
                raise InvalidOutputError(
                    "Specialist checkpoint must be JSON-serializable"
                ) from None
            return outcome
        raise InvalidOutputError("Specialist outcome must be typed and match the active run")

    async def cancel(self, run_id: str) -> None:
        active = self._active
        if (
            active is not None
            and active.context.run_id == run_id
            and (active.worker is None or not active.worker.done())
        ):
            active.cancel(self._close_timeout_s)

    async def aclose(self) -> None:
        self._closed = True
        active = self._active
        if active is not None:
            await self.cancel(active.context.run_id)
            deadline = (
                active.cancel_deadline.result()
                if active.cancel_deadline.done()
                else asyncio.get_running_loop().time() + self._close_timeout_s
            )
            with suppress(TimeoutError):
                async with asyncio.timeout_at(deadline):
                    await active.done.wait()


class InProcessDriver(Driver):
    """Resolve registered Python callables from operator-provided profiles."""

    def __init__(self, handlers: Mapping[str, SpecialistHandler] | None = None) -> None:
        self._handlers = dict(handlers or {})

    @asynccontextmanager
    async def open(
        self,
        profile: AgentProfile,
        *,
        workspace_path: Path | None,
        access: WorkspaceAccess,
        resume_ref: str | None = None,
    ) -> AsyncIterator[InProcessSession]:
        selector = profile.options.get("handler", profile.name)
        handler = self._handlers.get(selector) if isinstance(selector, str) else selector
        if not callable(handler):
            raise UnsupportedError("No in-process specialist is registered for this profile")
        declared = getattr(handler, "capabilities", profile.capabilities)
        if not isinstance(declared, Capabilities):
            raise UnsupportedError("Invalid in-process specialist capabilities")
        enforced = _READ_ONLY & declared.enforced_sandbox & profile.capabilities.enforced_sandbox
        if access not in enforced:
            raise UnsupportedError("The in-process specialist lacks the requested workspace scope")
        capabilities = declared.model_copy(update={"enforced_sandbox": enforced})
        if resume_ref is not None and not capabilities.resume:
            raise ResumeUnavailableError(
                "The specialist does not support native session references"
            )
        if (
            "handler" not in profile.options
            and selector in ("invoice-qc", "echo")
            and isinstance(handler, InvoiceQcSpecialist | _EchoSpecialist)
        ):
            warn(
                "Profile name matched a built-in demo specialist. Set options.handler explicitly.",
                stacklevel=3,
            )
        session = InProcessSession(
            handler, profile, workspace_path, access, resume_ref, capabilities
        )
        try:
            yield session
        finally:
            await session.aclose()


@contextmanager
def _open_csv(root: Path, path: str) -> Iterator[BinaryIO]:
    components = path.split("/")
    if "\x00" in path or any(part in ("", ".", "..") for part in components):
        raise WorkspaceViolationError("Expected a clean workspace-relative CSV path")
    if not _SAFE_OPEN_SUPPORTED:
        raise UnsupportedError("Workspace CSV input requires POSIX directory descriptors")
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    try:
        directory = os.open(root, directory_flags)
        try:
            for component in components[:-1]:
                child = os.open(component, directory_flags, dir_fd=directory)
                os.close(directory)
                directory = child
            file_fd = os.open(
                components[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory
            )
            try:
                if not stat.S_ISREG(os.fstat(file_fd).st_mode):
                    raise WorkspaceViolationError("Expected a regular workspace CSV file")
                with os.fdopen(file_fd, "rb", closefd=False) as file:
                    yield file
            finally:
                os.close(file_fd)
        finally:
            os.close(directory)
    except (OSError, ValueError):
        raise WorkspaceViolationError("Could not safely read the workspace CSV file") from None


async def _verified_csv(root: Path, ref: WorkspaceRef) -> bytearray:
    data = bytearray()
    digest = hashlib.sha256()
    with _open_csv(root, ref.path) as file:
        while chunk := file.read(1024 * 1024):
            if len(data) + len(chunk) > ref.size_bytes:
                raise WorkspaceViolationError("Workspace CSV byte count differs from its reference")
            digest.update(chunk)
            data.extend(chunk)
            await asyncio.sleep(0)
    if len(data) != ref.size_bytes or digest.hexdigest() != ref.sha256.lower():
        raise WorkspaceViolationError("Workspace CSV size or SHA-256 differs from its reference")
    return data


_InvoiceId = Annotated[str, Field(min_length=1, validation_alias=AliasChoices("invoice_id", "id"))]


class _InvoiceScore(BaseModel):
    invoice_id: _InvoiceId
    score: float = Field(ge=0, le=1, allow_inf_nan=False)


class _InvoiceAmounts(BaseModel):
    invoice_id: _InvoiceId
    amount: float = Field(ge=0, allow_inf_nan=False)
    expected_amount: float = Field(ge=0, allow_inf_nan=False)


class _CutoffCheckpoint(BaseModel):
    stage: Literal["awaiting_cutoff"]
    items: list[_InvoiceScore]
    input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def _score_invoice(row: Any) -> _InvoiceScore:
    if isinstance(row, dict) and "score" in row:
        return _InvoiceScore.model_validate(row)
    invoice = _InvoiceAmounts.model_validate(row)
    scale = max(invoice.amount, invoice.expected_amount)
    score = abs(invoice.amount - invoice.expected_amount) / scale if scale else 0.0
    return _InvoiceScore(invoice_id=invoice.invoice_id, score=score)


def _score_csv(text: str) -> list[_InvoiceScore]:
    reader = csv.DictReader(io.StringIO(text, newline=""), strict=True)
    headers = set(reader.fieldnames or ())
    has_amounts = {"amount", "expected_amount"} <= headers
    if not headers & {"invoice_id", "id"} or not ("score" in headers or has_amounts):
        raise ValueError("CSV requires invoice IDs and either scores or invoice amounts")
    if len(headers) != len(reader.fieldnames or ()):
        raise ValueError("CSV headers must be unique")
    items = []
    for row in reader:
        if None in row:
            raise ValueError("CSV row contains fields beyond the header")
        items.append(_score_invoice(row))
    return items


async def _checkpoint(context: SpecialistContext, request: TurnRequest) -> _CutoffCheckpoint:
    if request.input_refs:
        if context.workspace_path is None:
            raise WorkspaceViolationError("Workspace CSV references require a workspace binding")
        items = []
        digests = []
        for ref in request.input_refs:
            source = await _verified_csv(context.workspace_path, ref)
            items.extend(_score_csv(source.decode("utf-8-sig")))
            digests.append(bytes.fromhex(ref.sha256))
        digest = digests[0] if len(digests) == 1 else hashlib.sha256(b"".join(digests)).digest()
    else:
        prompt = request.prompt.removeprefix("\ufeff")
        if prompt.lstrip().startswith(("[", "{")):
            rows = json.loads(prompt)
            if isinstance(rows, dict):
                rows = rows.get("items")
            if not isinstance(rows, list):
                raise ValueError("Invoice JSON requires an array of rows")
            items = [_score_invoice(row) for row in rows]
        else:
            items = _score_csv(prompt)
        digest = hashlib.sha256(request.prompt.encode("utf-8")).digest()
    return _CutoffCheckpoint(stage="awaiting_cutoff", items=items, input_sha256=digest.hex())


class InvoiceQcSpecialist:
    """Score invoices and resume from the caller's JSON checkpoint."""

    capabilities = Capabilities(task_questions=True, thinking="final", enforced_sandbox=_READ_ONLY)

    async def __call__(self, context: SpecialistContext, request: TurnRequest) -> TurnOutcome:
        if request.continuation is None:
            await context.emit(
                ThoughtEvent, text="Score invoices before applying a cutoff.", is_delta=False
            )
            await context.emit(ToolEvent, call_id="score", name="invoice_qc", phase="started")
            try:
                checkpoint = await _checkpoint(context, request)
            except Exception:
                await context.emit(ToolEvent, call_id="score", name="invoice_qc", phase="failed")
                raise
            await context.emit(ToolEvent, call_id="score", name="invoice_qc", phase="completed")
            return TurnNeedsInput(
                run_id=request.run_id,
                question=_CUTOFF_QUESTION,
                continuation=checkpoint.model_dump(mode="json"),
            )

        checkpoint = _CutoffCheckpoint.model_validate(request.continuation)
        cutoff = None
        with suppress(ValueError):
            cutoff = float(request.prompt)
        if cutoff is None or not 0 <= cutoff <= 1:
            return TurnNeedsInput(
                run_id=request.run_id,
                question="Enter a numeric cutoff between 0 and 1.",
                continuation=request.continuation,
            )
        flagged = [row.invoice_id for row in checkpoint.items if row.score >= cutoff]
        total = len(checkpoint.items)
        text = f"Flagged {len(flagged)} of {total} invoices."
        await context.emit(TextEvent, text=text, is_delta=False)
        return TurnFinished(
            result=TurnResult(
                run_id=request.run_id,
                status="completed",
                text=text,
                has_structured=True,
                structured={
                    "cutoff": cutoff,
                    "flagged": flagged,
                    "total": total,
                    "passed": total - len(flagged),
                },
            )
        )


def make_cutoff_qc_specialist() -> InvoiceQcSpecialist:
    return InvoiceQcSpecialist()


class _EchoSpecialist:
    capabilities = Capabilities(approvals="live", enforced_sandbox=_READ_ONLY)

    async def __call__(self, context: SpecialistContext, request: TurnRequest) -> TurnOutcome:
        action = context.profile.options.get("approval_action")
        if action is not None:
            decision = await context.request_approval(
                action, "Allow the specialist to echo the prompt?"
            )
            if decision.outcome != "allow_once":
                status = "failed" if decision.outcome == "deny" else "cancelled"
                return TurnFinished(
                    result=TurnResult(
                        run_id=request.run_id, status=status, reason="approval_denied"
                    )
                )
        if context.profile.options.get("slow") is True or request.prompt == "slow":
            await context.cancelled.wait()
        await context.emit(TextEvent, text=request.prompt, is_delta=False)
        return TurnFinished(
            result=TurnResult(run_id=request.run_id, status="completed", text=request.prompt)
        )


def make_echo_specialist() -> SpecialistHandler:
    """Make an echo handler with optional approval_action and slow profile options."""
    return _EchoSpecialist()
