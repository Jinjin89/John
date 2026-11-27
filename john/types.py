"""Frozen internal contract types for John's headless runtime layer.

This module intentionally contains zero backend-specific identifiers or vendor SDK imports.
Outer inter-agent envelopes use A2A v1.0 protobuf types (`a2a.types`) directly.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from contextlib import AbstractAsyncContextManager
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

WORKSPACE_EXT_URI = "https://john.dev/extensions/workspace/v1"
WORKSPACE_REF_MEDIA_TYPE = "application/vnd.john.workspace-ref+json"
LIFECYCLE_EXT_URI = "https://john.dev/extensions/lifecycle/v1"

WorkspaceAccess = Literal["read_only", "read_write"]
ApprovalMode = Literal["live", "auto_only", "none"]
ThinkingMode = Literal["stream", "final", "hidden"]
ApprovalOutcome = Literal["allow_once", "deny", "deny_cancel", "cancelled"]
TurnStatus = Literal["completed", "incomplete", "failed", "cancelled", "indeterminate"]


def utc_now_iso() -> str:
    """Return an ISO-8601 UTC timestamp with second precision."""
    return datetime.now(UTC).isoformat()


# ---------------------------------------------------------------------------
# Closed Error Taxonomy
# ---------------------------------------------------------------------------


class JohnError(Exception):
    """Base error for all John runtime and contract failures."""

    code: str = "john_error"


class UnsupportedError(JohnError):
    """Raised before dispatch when a required capability or scope cannot be honored."""

    code = "unsupported"


class BusyError(JohnError):
    """Raised when a second concurrent turn is submitted to an active session."""

    code = "busy"


class AuthRequiredError(JohnError):
    """Raised when a harness lacks required credentials."""

    code = "auth_required"


class WorkspaceViolationError(JohnError):
    """Raised on unauthorized workspace roots, invalid paths, traversal, or digest mismatch."""

    code = "workspace_violation"


class InvalidOutputError(JohnError):
    """Raised when an agent's structured output or declared file output fails validation."""

    code = "invalid_output"


class ResumeUnavailableError(JohnError):
    """Raised when an explicit resume target is missing or incompatible with the current profile."""

    code = "resume_unavailable"


class TransportError(JohnError):
    """Raised on subprocess launch failure, malformed wire frames, or pipe errors."""

    code = "transport_error"


class DuplicateMessageConflictError(JohnError):
    """Raised when a message_id is reused with a different request fingerprint."""

    code = "duplicate_message_conflict"


# ---------------------------------------------------------------------------
# Workspace & Profile Models
# ---------------------------------------------------------------------------


class OutputSpec(BaseModel):
    """Declares an expected workspace-relative output file for a file-producing task."""

    model_config = ConfigDict(frozen=True, populate_by_name=True)

    name: str = Field(min_length=1)
    path: str = Field(min_length=1)
    media_type: str = Field(default="application/octet-stream", alias="mediaType")


class WorkspaceRequest(BaseModel):
    """Task-level workspace scope requested via the A2A workspace extension."""

    model_config = ConfigDict(frozen=True)

    root: str = Field(min_length=1)
    access: WorkspaceAccess = "read_only"
    outputs: tuple[OutputSpec, ...] = ()


class WorkspaceRef(BaseModel):
    """Verified workspace-relative artifact reference (paths + digests, never raw bytes)."""

    model_config = ConfigDict(frozen=True, populate_by_name=True)

    root: str = Field(min_length=1)
    path: str = Field(min_length=1)
    media_type: str = Field(alias="mediaType")
    size_bytes: int = Field(ge=0, alias="sizeBytes")
    sha256: str = Field(min_length=64, max_length=64)


class WorkspaceGrant(BaseModel):
    """Authorized host workspace root registered in John."""

    model_config = ConfigDict(frozen=True)

    root_id: str = Field(min_length=1)
    host_path: Path
    max_access: WorkspaceAccess = "read_only"
    allowed_principals: frozenset[str] = frozenset()
    allowed_agents: frozenset[str] = frozenset()


class Capabilities(BaseModel):
    """Declared or verified capability bounds of a driver or active session."""

    model_config = ConfigDict(frozen=True)

    approvals: ApprovalMode = "none"
    streaming: bool = True
    resume: bool = False
    thinking: ThinkingMode = "hidden"
    native_schema: bool = False
    enforced_sandbox: frozenset[WorkspaceAccess] = frozenset()
    task_questions: bool = False


class SkillSpec(BaseModel):
    """Skill metadata published into an agent's A2A AgentCard."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    description: str = ""
    tags: tuple[str, ...] = ()
    input_modes: tuple[str, ...] = ("text/plain",)
    output_modes: tuple[str, ...] = ("text/plain",)


class AgentProfile(BaseModel):
    """Configuration profile for one local or remote agent (data-driven from agents.toml)."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(min_length=1)
    kind: str = Field(min_length=1)
    description: str = ""
    version: str = "0.1.0"
    command: str | None = None
    args: tuple[str, ...] = ()
    env: dict[str, str] = Field(default_factory=dict)
    url: str | None = None
    default_access: WorkspaceAccess = "read_only"
    require_workspace: bool = False
    profile_dir: Path | None = None
    skills: tuple[SkillSpec, ...] = ()
    default_input_modes: tuple[str, ...] = ("text/plain",)
    default_output_modes: tuple[str, ...] = ("text/plain", "application/json")
    capabilities: Capabilities = Field(default_factory=Capabilities)
    options: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Approvals
# ---------------------------------------------------------------------------


class ApprovalRequest(BaseModel):
    """Inbound mid-turn tool permission request handled inside a local bridge."""

    model_config = ConfigDict(frozen=True)

    request_id: str
    run_id: str
    action: str
    summary: str = ""
    args: dict[str, Any] = Field(default_factory=dict)
    offered_options: tuple[str, ...] = ()


class ApprovalDecision(BaseModel):
    """Normalized decision returned by an ApprovalPolicy."""

    model_config = ConfigDict(frozen=True)

    outcome: ApprovalOutcome = "deny"
    reason: str = ""


ApprovalPolicy = Callable[[ApprovalRequest], Awaitable[ApprovalDecision]]


async def deny_all(req: ApprovalRequest) -> ApprovalDecision:
    """Default unattended approval policy: deny every tool permission prompt."""
    return ApprovalDecision(outcome="deny", reason=f"Denied by default policy: {req.action}")


# ---------------------------------------------------------------------------
# Closed Canonical Event Union (Private Runtime Stream & Run Journal)
# ---------------------------------------------------------------------------


class _BaseEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    run_id: str
    seq: int = Field(ge=0)
    timestamp: str = Field(default_factory=utc_now_iso)


class StartedEvent(_BaseEvent):
    kind: Literal["started"] = "started"
    native_session_ref: str | None = None
    ignored_fields: dict[str, str] = Field(default_factory=dict)


class TextEvent(_BaseEvent):
    kind: Literal["text"] = "text"
    text: str
    is_delta: bool = True


class ThoughtEvent(_BaseEvent):
    kind: Literal["thought"] = "thought"
    text: str
    is_delta: bool = True


class ToolEvent(_BaseEvent):
    kind: Literal["tool"] = "tool"
    call_id: str
    name: str
    phase: Literal["started", "updated", "completed", "failed"]
    summary: str = ""


class PlanEvent(_BaseEvent):
    kind: Literal["plan"] = "plan"
    items: tuple[str, ...] = ()


class ApprovalEvent(_BaseEvent):
    kind: Literal["approval"] = "approval"
    request_id: str
    action: str
    phase: Literal["requested", "decided"]
    outcome: ApprovalOutcome | None = None
    reason: str = ""


class UsageEvent(_BaseEvent):
    kind: Literal["usage"] = "usage"
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


class ErrorEvent(_BaseEvent):
    kind: Literal["error"] = "error"
    code: str
    message: str
    retryable: bool = False


class OtherEvent(_BaseEvent):
    """Preserves unrecognized native frames without silent drops; raw payload is opt-in."""

    kind: Literal["other"] = "other"
    native_type: str
    raw: dict[str, Any] | None = None
    redacted: bool = True


class FinishedEvent(_BaseEvent):
    kind: Literal["finished"] = "finished"
    status: TurnStatus
    reason: str = ""


RuntimeEvent = Annotated[
    StartedEvent
    | TextEvent
    | ThoughtEvent
    | ToolEvent
    | PlanEvent
    | ApprovalEvent
    | UsageEvent
    | ErrorEvent
    | OtherEvent
    | FinishedEvent,
    Field(discriminator="kind"),
]


# ---------------------------------------------------------------------------
# Turn Requests, Outcomes & Driver Protocols (Inner SPI)
# ---------------------------------------------------------------------------


class TurnRequest(BaseModel):
    """One native turn dispatched to an open DriverSession."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    prompt: str
    input_refs: tuple[WorkspaceRef, ...] = ()
    outputs: tuple[OutputSpec, ...] = ()
    schema_def: dict[str, Any] | None = None
    continuation: dict[str, Any] | None = None
    timeout_s: float | None = None


class TurnResult(BaseModel):
    """Terminal result produced by a single native turn."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    status: TurnStatus
    reason: str = ""
    text: str = ""
    structured: Any = None
    has_structured: bool = False
    usage: dict[str, Any] | None = None
    native_session_ref: str | None = None


class TurnFinished(BaseModel):
    """Driver outcome indicating the turn reached a terminal state."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["finished"] = "finished"
    result: TurnResult


class TurnNeedsInput(BaseModel):
    """Driver outcome indicating a task-level question (maps to A2A INPUT_REQUIRED)."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["needs_input"] = "needs_input"
    run_id: str
    question: str
    continuation: dict[str, Any] = Field(default_factory=dict)
    native_session_ref: str | None = None


TurnOutcome = Annotated[TurnFinished | TurnNeedsInput, Field(discriminator="kind")]

EventSink = Callable[[RuntimeEvent], Awaitable[None]]


@runtime_checkable
class DriverSession(Protocol):
    """Active handle to one opened harness/graph session."""

    @property
    def native_ref(self) -> str | None: ...

    @property
    def capabilities(self) -> Capabilities: ...

    @property
    def ignored_fields(self) -> dict[str, str]: ...

    async def run_turn(
        self,
        request: TurnRequest,
        on_event: EventSink,
        on_approval: ApprovalPolicy,
    ) -> TurnOutcome: ...

    async def cancel(self, run_id: str) -> None: ...


@runtime_checkable
class Driver(Protocol):
    """Factory that opens a DriverSession bound to a profile, workspace, and optional resume ref."""

    def open(
        self,
        profile: AgentProfile,
        *,
        workspace_path: Path | None,
        access: WorkspaceAccess,
        resume_ref: str | None = None,
    ) -> AbstractAsyncContextManager[DriverSession]: ...
