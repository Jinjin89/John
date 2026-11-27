"""Execute one local driver turn and journal its canonical events."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from itertools import count
from typing import Any

import jsonschema

from john.ledger import SqliteLedger
from john.transport import redact_text
from john.types import (
    ApprovalDecision,
    ApprovalEvent,
    ApprovalPolicy,
    ApprovalRequest,
    BusyError,
    DriverSession,
    EventSink,
    FinishedEvent,
    RuntimeEvent,
    TransportError,
    TurnFinished,
    TurnNeedsInput,
    TurnOutcome,
    TurnRequest,
    TurnResult,
    TurnStatus,
    WorkspaceAccess,
    deny_all,
)

_LOG = logging.getLogger(__name__)
_CANCELLED_APPROVAL = ApprovalDecision(outcome="cancelled", reason="turn_cancelled")

type _TaskKey = tuple[str, str, str]
type _TurnCompletion = tuple[TurnOutcome, int]


def make_seq_allocator() -> Callable[[], int]:
    """Return a fresh sequence allocator for one run."""
    return count().__next__


def _retrieve_exception(task: asyncio.Future[Any]) -> None:
    if not task.cancelled():
        task.exception()


class ApprovalBroker:
    """Journal approval requests and resolve each request ID once per turn."""

    def __init__(
        self,
        policy: ApprovalPolicy = deny_all,
        timeout_s: float = 30.0,
        emit_event: EventSink | None = None,
    ) -> None:
        self._policy = policy
        self._timeout_s = timeout_s
        self._emit_event = emit_event
        self._seq = make_seq_allocator()
        self._resolved: dict[str, tuple[str, str, asyncio.Task[ApprovalDecision]]] = {}
        self._waits: set[asyncio.Future[ApprovalDecision]] = set()
        self._cancelled = False

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    def mark_cancelled(self) -> None:
        self._cancelled = True
        for wait in self._waits:
            wait.cancel()

    async def decide(self, req: ApprovalRequest) -> ApprovalDecision:
        try:
            args = json.dumps(req.args, sort_keys=True, allow_nan=False)
        except (TypeError, ValueError, RecursionError):
            return ApprovalDecision(reason="invalid_approval_request_args")
        previous = self._resolved.get(req.request_id)
        if previous is None:
            resolution = asyncio.create_task(self._resolve(req.model_copy(deep=True)))
            resolution.add_done_callback(_retrieve_exception)
            self._resolved[req.request_id] = (req.action, args, resolution)
        else:
            action, original_args, resolution = previous
            if (req.action, args) != (action, original_args):
                return ApprovalDecision(reason="conflicting_approval_request_id")
        decision = await asyncio.shield(resolution)
        return _CANCELLED_APPROVAL if self._cancelled else decision

    async def _resolve(self, req: ApprovalRequest) -> ApprovalDecision:
        fields = {
            "run_id": req.run_id,
            "request_id": req.request_id,
            "action": redact_text(req.action),
        }
        if self._emit_event is not None:
            await self._emit_event(ApprovalEvent(**fields, seq=self._seq(), phase="requested"))

        decision = _CANCELLED_APPROVAL if self._cancelled else await self._wait_for_policy(req)
        if self._cancelled:
            decision = _CANCELLED_APPROVAL
        if self._emit_event is not None:
            await self._emit_event(
                ApprovalEvent(
                    **fields,
                    seq=self._seq(),
                    phase="decided",
                    outcome=decision.outcome,
                    reason=redact_text(decision.reason),
                )
            )
        return decision

    async def _wait_for_policy(self, req: ApprovalRequest) -> ApprovalDecision:
        async def invoke() -> ApprovalDecision:
            decision = await self._policy(req)
            if not isinstance(decision, ApprovalDecision):
                raise TypeError("Approval policy returned an invalid decision")
            return decision

        policy_task = asyncio.create_task(invoke())
        policy_task.add_done_callback(_retrieve_exception)
        # Cancel the wait independently. A policy that catches cancellation
        # cannot extend the deadline or supply a late decision.
        wait = asyncio.shield(policy_task)
        self._waits.add(wait)
        try:
            return await asyncio.wait_for(wait, timeout=self._timeout_s)
        except TimeoutError:
            return ApprovalDecision(reason="approval_timeout")
        except asyncio.CancelledError:
            if self._cancelled:
                return _CANCELLED_APPROVAL
            if (task := asyncio.current_task()) is not None and task.cancelling():
                raise
            return ApprovalDecision(reason="policy_error: CancelledError")
        except Exception as exc:
            return ApprovalDecision(reason=f"policy_error: {type(exc).__name__}")
        finally:
            self._waits.discard(wait)
            if not policy_task.done():
                policy_task.cancel()

    async def _close(self) -> None:
        self.mark_cancelled()
        await asyncio.gather(
            *(resolution for _, _, resolution in self._resolved.values()), return_exceptions=True
        )


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"{value} is not a JSON number")


def _has_unsafe_integer(value: Any) -> bool:
    if isinstance(value, int) and not isinstance(value, bool):
        return abs(value) > (1 << 53) - 1
    if isinstance(value, dict):
        return any(_has_unsafe_integer(item) for item in value.values())
    if isinstance(value, list | tuple):
        return any(_has_unsafe_integer(item) for item in value)
    return False


def validate_json_output(result: TurnResult, schema_def: dict[str, Any]) -> TurnResult:
    """Validate a completed result, including a structured JSON null."""
    if result.status != "completed":
        return result
    try:
        candidate = (
            result.structured
            if result.has_structured
            else json.loads(result.text, parse_constant=_reject_json_constant)
        )
        if _has_unsafe_integer(candidate):
            reason = "invalid_output: integer exceeds JSON/protobuf exact number precision"
            return result.model_copy(
                update={
                    "status": "failed",
                    "reason": reason,
                }
            )
        json.dumps(candidate, allow_nan=False)
    except (ValueError, TypeError, RecursionError):
        return result.model_copy(update={"status": "failed", "reason": "invalid_output"})
    try:
        jsonschema.validate(instance=candidate, schema=schema_def)
    except jsonschema.ValidationError as exc:
        reason = f"invalid_output: schema constraint {exc.validator!r} failed"
        return result.model_copy(update={"status": "failed", "reason": reason})
    except jsonschema.SchemaError:
        return result.model_copy(update={"status": "failed", "reason": "invalid_schema"})
    return result.model_copy(update={"structured": candidate, "has_structured": True})


class _RunJournal:
    def __init__(
        self,
        ledger: SqliteLedger,
        run_id: str,
        observations: asyncio.Queue[RuntimeEvent | None] | None,
        observer: asyncio.Task[None] | None,
    ) -> None:
        self._ledger = ledger
        self._run_id = run_id
        self._observations = observations
        self._observer = observer
        self._seq = make_seq_allocator()
        self._lock = asyncio.Lock()
        self._closed = False
        self._seen_approvals: set[tuple[str, str]] = set()

    async def emit(self, event: RuntimeEvent) -> None:
        # The validated outcome supplies the single terminal ledger event.
        if isinstance(event, FinishedEvent):
            return
        async with self._lock:
            if self._closed:
                return
            if isinstance(event, ApprovalEvent):
                approval_key = (event.request_id, event.phase)
                if approval_key in self._seen_approvals:
                    return
                self._seen_approvals.add(approval_key)
            record = event.model_copy(update={"run_id": self._run_id, "seq": self._seq()})
            await self._ledger.append_event(record)
            self._offer(record)

    def _offer(self, event: RuntimeEvent | None) -> None:
        queue = self._observations
        if queue is None:
            return
        try:
            queue.put_nowait(event)
        except asyncio.QueueFull:
            self._observations = None
            _LOG.warning("Runtime event observer disconnected after its queue filled")
            while not queue.empty():
                queue.get_nowait()
            queue.put_nowait(None)
            if self._observer is not None:
                self._observer.cancel()

    async def seal(self) -> int:
        async with self._lock:
            self._closed = True
            self._offer(None)
            return self._seq()


async def _observe_events(queue: asyncio.Queue[RuntimeEvent | None], sink: EventSink) -> None:
    while (event := await queue.get()) is not None:
        try:
            await sink(event)
        except Exception as exc:
            _LOG.warning("Runtime event observer failed (%s)", type(exc).__name__)


@dataclass(slots=True)
class _ActiveTurn:
    session: DriverSession
    run_id: str
    context_id: str
    broker: ApprovalBroker
    completion: asyncio.Future[_TurnCompletion]
    abandoned: asyncio.Future[str]
    native: asyncio.Task[TurnOutcome] | None = None
    cancellation: asyncio.Task[None] | None = None


def _finished(active: _ActiveTurn, status: TurnStatus, reason: str) -> TurnFinished:
    return TurnFinished(
        result=TurnResult(
            run_id=active.run_id,
            status=status,
            reason=reason,
            native_session_ref=active.session.native_ref,
        )
    )


class SessionRuntime:
    """Own active turns and serialize their journal writes without blocking approvals."""

    def __init__(self, ledger: SqliteLedger, *, cancel_timeout_s: float = 10.0) -> None:
        self.ledger = ledger
        self._cancel_timeout_s = cancel_timeout_s
        self._active: dict[_TaskKey, _ActiveTurn] = {}
        self._background: set[asyncio.Task[Any]] = set()

    def _spawn[T](self, coro: Coroutine[Any, Any, T]) -> asyncio.Task[T]:
        task = asyncio.create_task(coro)
        self._background.add(task)
        task.add_done_callback(self._background.discard)
        task.add_done_callback(_retrieve_exception)
        return task

    async def execute_turn(
        self,
        session: DriverSession,
        request: TurnRequest,
        *,
        owner: str,
        agent_name: str,
        task_id: str,
        context_id: str,
        policy: ApprovalPolicy = deny_all,
        approval_timeout_s: float = 30.0,
        parent_run_id: str | None = None,
        workspace_root_id: str | None = None,
        workspace_access: WorkspaceAccess | None = None,
        continuation_run_id: str | None = None,
        on_event: EventSink | None = None,
    ) -> _TurnCompletion:
        key = (owner, agent_name, task_id)
        if key in self._active or any(
            turn.session is session or (scope[:2], turn.context_id) == (key[:2], context_id)
            for scope, turn in self._active.items()
        ):
            raise BusyError("The session already has an active turn")

        observations = (
            asyncio.Queue[RuntimeEvent | None](maxsize=256) if on_event is not None else None
        )
        observer = (
            self._spawn(_observe_events(observations, on_event))
            if observations is not None and on_event is not None
            else None
        )
        journal = _RunJournal(self.ledger, request.run_id, observations, observer)
        broker = ApprovalBroker(policy, approval_timeout_s, journal.emit)
        loop = asyncio.get_running_loop()
        active = _ActiveTurn(
            session, request.run_id, context_id, broker, loop.create_future(), loop.create_future()
        )
        self._active[key] = active
        outcome = None
        try:
            await self.ledger.start_run(
                request.run_id,
                owner=owner,
                agent_name=agent_name,
                task_id=task_id,
                context_id=context_id,
                parent_run_id=parent_run_id,
                workspace_root_id=workspace_root_id,
                workspace_access=workspace_access,
                continuation_run_id=continuation_run_id,
            )
            if broker.cancelled:
                outcome = _finished(active, "cancelled", "cancelled_before_dispatch")
            else:
                outcome = await self._run_native(active, request, journal.emit)
        except asyncio.CancelledError:
            stop = self._request_cancel(active, self._cancel_timeout_s, "cancel_timeout")
            await asyncio.shield(stop)
            raise
        finally:
            # The caller may be canceled during finalization as well as during
            # native execution. Let cleanup finish independently in both cases.
            cleanup = self._spawn(self._complete_turn(key, active, journal, outcome))
            next_seq = await asyncio.shield(cleanup)
        return outcome, next_seq

    async def _complete_turn(
        self, key: _TaskKey, active: _ActiveTurn, journal: _RunJournal, outcome: TurnOutcome | None
    ) -> int:
        try:
            await active.broker._close()
            next_seq = await journal.seal()
            if outcome is not None:
                active.completion.set_result((outcome, next_seq))
            return next_seq
        finally:
            if not active.completion.done():
                active.completion.cancel()
            active.abandoned.cancel()
            if active.native is not None and not active.native.done():
                # A cancellation-resistant driver still owns the session until it exits.
                active.native.add_done_callback(lambda _: self._active.pop(key))
            else:
                del self._active[key]

    async def _run_native(
        self, active: _ActiveTurn, request: TurnRequest, emit: EventSink
    ) -> TurnOutcome:
        async def invoke() -> TurnOutcome:
            return await active.session.run_turn(request, emit, active.broker.decide)

        native = active.native = self._spawn(invoke())
        done, _ = await asyncio.wait(
            (native, active.abandoned),
            timeout=request.timeout_s,
            return_when=asyncio.FIRST_COMPLETED,
        )
        if not done:
            stop = self._request_cancel(active, self._cancel_timeout_s, "turn_timeout")
            await asyncio.shield(stop)
        if active.abandoned.done():
            return _finished(active, "indeterminate", active.abandoned.result())

        try:
            outcome = native.result()
            if isinstance(outcome, TurnNeedsInput) and outcome.run_id == request.run_id:
                if not active.session.capabilities.task_questions:
                    return _finished(active, "failed", "unsupported_task_question")
                return outcome
            if isinstance(outcome, TurnFinished) and outcome.result.run_id == request.run_id:
                if request.schema_def is not None or outcome.result.has_structured:
                    return TurnFinished(
                        result=validate_json_output(outcome.result, request.schema_def or {})
                    )
                return outcome
            return _finished(active, "indeterminate", "invalid_driver_outcome")
        except asyncio.CancelledError:
            return _finished(active, "indeterminate", "driver_cancelled_without_confirmation")
        except TransportError:
            return _finished(active, "indeterminate", TransportError.code)
        except Exception as exc:
            return _finished(active, "indeterminate", f"driver_error: {type(exc).__name__}")

    async def cancel_task(
        self,
        task_id: str,
        timeout_s: float = 10.0,
        *,
        owner: str | None = None,
        agent_name: str | None = None,
        run_id: str | None = None,
    ) -> _TurnCompletion | None:
        """Request native cancellation and wait for a confirmed or indeterminate outcome."""
        matches = [
            active
            for (task_owner, task_agent, active_task_id), active in self._active.items()
            if active_task_id == task_id
            and (owner is None or owner == task_owner)
            and (agent_name is None or agent_name == task_agent)
            and (run_id is None or run_id == active.run_id)
        ]
        if not matches:
            return None
        if len(matches) != 1:
            raise BusyError("Task ID has active turns in multiple scopes")
        (active,) = matches
        self._request_cancel(active, timeout_s, "cancel_timeout")
        return await asyncio.shield(active.completion)

    def _request_cancel(
        self, active: _ActiveTurn, timeout_s: float, reason: str
    ) -> asyncio.Task[None]:
        active.broker.mark_cancelled()
        if active.cancellation is None:
            active.cancellation = self._spawn(self._cancel_native(active, timeout_s, reason))
        return active.cancellation

    async def _cancel_native(self, active: _ActiveTurn, timeout_s: float, reason: str) -> None:
        native = active.native
        if native is None or native.done():
            return
        command = self._spawn(active.session.cancel(active.run_id))
        try:
            await asyncio.wait((native,), timeout=timeout_s)
            if not native.done():
                if not active.abandoned.done():
                    active.abandoned.set_result(reason)
                native.cancel()
        finally:
            if not command.done():
                command.cancel()
