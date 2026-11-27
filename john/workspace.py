"""Authorize local workspaces and exchange verified A2A file references."""

from __future__ import annotations

import hashlib
import io
import os
import stat
from collections.abc import Iterator, Sequence
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from tempfile import TemporaryFile
from typing import Any, BinaryIO

from a2a.helpers.proto_helpers import new_data_part
from a2a.types import Artifact
from google.protobuf.json_format import MessageToDict
from pydantic import ValidationError

from john.types import (
    WORKSPACE_EXT_URI,
    WORKSPACE_REF_MEDIA_TYPE,
    Capabilities,
    OutputSpec,
    UnsupportedError,
    WorkspaceAccess,
    WorkspaceGrant,
    WorkspaceRef,
    WorkspaceRequest,
    WorkspaceViolationError,
)

_CHUNK_SIZE = 1024 * 1024
_SAFE_OPEN_SUPPORTED = os.open in os.supports_dir_fd and all(
    hasattr(os, flag) for flag in ("O_DIRECTORY", "O_NOFOLLOW", "O_NONBLOCK")
)


class ExtensionRequiredError(UnsupportedError):
    """A required workspace extension was not activated for the request."""


class WorkspaceRegistry:
    """Workspace grants configured by the host operator."""

    def __init__(self) -> None:
        self._grants: dict[str, WorkspaceGrant] = {}

    def register(self, grant: WorkspaceGrant) -> None:
        if not grant.host_path.is_absolute():
            grant = grant.model_copy(update={"host_path": grant.host_path.absolute()})
        self._grants[grant.root_id] = grant

    def get_grant(self, root_id: str) -> WorkspaceGrant:
        try:
            return self._grants[root_id]
        except KeyError:
            raise WorkspaceViolationError("Unknown workspace root") from None

    def authorize(
        self,
        root_id: str,
        access: WorkspaceAccess,
        *,
        principal: str = "local",
        agent_name: str = "",
    ) -> WorkspaceGrant:
        grant = self._authorize_scope(root_id, access, principal, agent_name)
        with _open_root_directory(grant.host_path):
            return grant

    def _authorize_scope(
        self, root_id: str, access: WorkspaceAccess, principal: str, agent_name: str
    ) -> WorkspaceGrant:
        grant = self.get_grant(root_id)
        if grant.allowed_principals and principal not in grant.allowed_principals:
            raise WorkspaceViolationError("Principal is unauthorized for this workspace")
        if grant.allowed_agents and agent_name not in grant.allowed_agents:
            raise WorkspaceViolationError("Agent is unauthorized for this workspace")
        if access not in ("read_only", "read_write"):
            raise WorkspaceViolationError("Invalid workspace access")
        if access == "read_write" and grant.max_access == "read_only":
            raise WorkspaceViolationError("Workspace grant allows read-only access")
        return grant

    def _open_for_read(
        self, root_id: str, path: str, principal: str, agent_name: str
    ) -> AbstractContextManager[tuple[BinaryIO, os.stat_result]]:
        grant = self._authorize_scope(root_id, "read_only", principal, agent_name)
        return open_safe_regular_file(grant.host_path, path)


def parse_workspace_request(
    metadata: dict[str, Any] | None,
    requested_extensions: set[str],
    *,
    required: bool,
) -> WorkspaceRequest | None:
    """Parse the workspace metadata from an A2A message."""
    active = WORKSPACE_EXT_URI in requested_extensions
    if required and not active:
        raise ExtensionRequiredError("Activate the required workspace extension")
    if metadata is not None and not isinstance(metadata, dict):
        raise WorkspaceViolationError("Expected an A2A message metadata object")

    fields = metadata or {}
    if not active:
        if WORKSPACE_EXT_URI in fields:
            raise WorkspaceViolationError("Activate the workspace extension to use its metadata")
        return None

    payload = fields.get(WORKSPACE_EXT_URI)
    if not isinstance(payload, dict):
        raise WorkspaceViolationError("Expected a workspace metadata object")
    try:
        request = WorkspaceRequest.model_validate(payload)
    except ValidationError:
        raise WorkspaceViolationError("Invalid workspace request metadata") from None
    _validate_outputs(request.outputs)
    return request


def validate_workspace_request(
    req: WorkspaceRequest,
    registry: WorkspaceRegistry,
    caps: Capabilities,
    *,
    principal: str = "local",
    agent_name: str = "",
) -> WorkspaceGrant:
    """Validate the grant, enforceable access, and declared outputs before dispatch."""
    grant = registry.authorize(req.root, req.access, principal=principal, agent_name=agent_name)
    if req.access not in caps.enforced_sandbox:
        raise UnsupportedError(
            "Driver lacks sandbox enforcement for the requested workspace access"
        )
    if req.outputs and req.access != "read_write":
        raise WorkspaceViolationError("Declared file outputs require read-write access")
    _validate_outputs(req.outputs)
    return grant


def _path_components(path: str) -> list[str]:
    components = path.split("/")
    if "\x00" in path or any(component in ("", ".", "..") for component in components):
        raise WorkspaceViolationError("Expected a clean workspace-relative POSIX path")
    return components


def _validate_outputs(outputs: tuple[OutputSpec, ...]) -> None:
    names: set[str] = set()
    for output in outputs:
        _path_components(output.path)
        if output.name in names:
            raise WorkspaceViolationError("Workspace output names must be unique")
        names.add(output.name)


@contextmanager
def _open_root_directory(root_path: Path) -> Iterator[int]:
    if not _SAFE_OPEN_SUPPORTED:
        raise UnsupportedError("Safe workspace opens require POSIX directory descriptors")
    try:
        root_fd = os.open(root_path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except (OSError, ValueError):
        raise WorkspaceViolationError("Expected an accessible real workspace directory") from None
    try:
        yield root_fd
    finally:
        os.close(root_fd)


@contextmanager
def _open_parent_directory(root_fd: int, components: Sequence[str]) -> Iterator[int]:
    parent_fd = root_fd
    try:
        for component in components:
            next_fd = os.open(
                component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent_fd
            )
            if parent_fd != root_fd:
                os.close(parent_fd)
            parent_fd = next_fd
        yield parent_fd
    finally:
        if parent_fd != root_fd:
            os.close(parent_fd)


def _open_leaf(root_fd: int, components: list[str]) -> int:
    try:
        with _open_parent_directory(root_fd, components[:-1]) as parent_fd:
            # A FIFO must not block this open before fstat can reject it.
            return os.open(
                components[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent_fd
            )
    except (OSError, ValueError):
        raise WorkspaceViolationError("Could not open workspace-relative path safely") from None


def assert_outputs_absent(
    root_id: str,
    outputs: Sequence[OutputSpec],
    registry: WorkspaceRegistry,
    *,
    principal: str = "local",
    agent_name: str = "",
) -> None:
    """Require fresh output paths within an authorized writable workspace."""
    grant = registry._authorize_scope(root_id, "read_write", principal, agent_name)
    with _open_root_directory(grant.host_path) as root_fd:
        for output in outputs:
            components = _path_components(output.path)
            try:
                with _open_parent_directory(root_fd, components[:-1]) as parent_fd:
                    os.stat(components[-1], dir_fd=parent_fd, follow_symlinks=False)
            except FileNotFoundError:
                continue
            except (OSError, ValueError):
                raise WorkspaceViolationError(
                    "Could not check declared output path safely"
                ) from None
            raise WorkspaceViolationError("Declared output path already exists")


@contextmanager
def open_safe_regular_file(
    root_path: Path, rel_path: str
) -> Iterator[tuple[BinaryIO, os.stat_result]]:
    """Yield the opened binary file and its fstat result beneath a pinned root."""
    components = _path_components(rel_path)
    with _open_root_directory(root_path) as root_fd:
        file_fd = _open_leaf(root_fd, components)
        try:
            try:
                file_stat = os.fstat(file_fd)
                if not stat.S_ISREG(file_stat.st_mode):
                    raise WorkspaceViolationError("Expected a regular workspace file")
                file = os.fdopen(file_fd, "rb", closefd=False)
            except OSError:
                raise WorkspaceViolationError("Could not read workspace file descriptor") from None
            with file:
                yield file, file_stat
        finally:
            os.close(file_fd)


def _hash_contents(
    source: BinaryIO, *, destination: BinaryIO | None = None, size_limit: int | None = None
) -> tuple[int, bytes]:
    digest = hashlib.sha256()
    size = 0
    try:
        for chunk in iter(lambda: source.read(_CHUNK_SIZE), b""):
            size += len(chunk)
            if size_limit is not None and size > size_limit:
                raise WorkspaceViolationError("Workspace artifact size mismatch")
            digest.update(chunk)
            if destination is not None:
                destination.write(chunk)
    except OSError:
        raise WorkspaceViolationError("Could not read or stage workspace contents") from None
    return size, digest.digest()


def build_workspace_artifact(
    root_id: str,
    spec: OutputSpec,
    registry: WorkspaceRegistry,
    *,
    principal: str = "local",
    agent_name: str = "",
) -> Artifact:
    """Hash an authorized output and publish its A2A workspace reference."""
    with registry._open_for_read(root_id, spec.path, principal, agent_name) as (file, _):
        size, digest = _hash_contents(file)
    if float(size) != size:
        raise WorkspaceViolationError("Workspace byte count exceeds protobuf number precision")
    ref = WorkspaceRef(
        root=root_id,
        path=spec.path,
        media_type=spec.media_type,
        size_bytes=size,
        sha256=digest.hex(),
    )
    return Artifact(
        artifact_id=spec.name,
        name=spec.name,
        extensions=[WORKSPACE_EXT_URI],
        parts=[new_data_part(ref.model_dump(by_alias=True), media_type=WORKSPACE_REF_MEDIA_TYPE)],
    )


def _extract_workspace_ref(artifact: Artifact) -> WorkspaceRef:
    parts = [part for part in artifact.parts if part.media_type == WORKSPACE_REF_MEDIA_TYPE]
    if len(parts) != 1 or parts[0].WhichOneof("content") != "data":
        raise WorkspaceViolationError("Expected exactly one workspace reference data part")
    try:
        data = MessageToDict(parts[0].data)
        if not isinstance(data, dict):
            raise WorkspaceViolationError("Expected a workspace reference object")
        size = data.get("sizeBytes", data.get("size_bytes"))
        if type(size) not in (int, float):
            raise WorkspaceViolationError("Workspace reference size must be a number")
        return WorkspaceRef.model_validate(data)
    except ValueError:
        raise WorkspaceViolationError("Invalid workspace reference data") from None


def _verify_contents(
    ref: WorkspaceRef,
    registry: WorkspaceRegistry,
    *,
    principal: str,
    agent_name: str,
    destination: BinaryIO | None = None,
) -> None:
    try:
        expected_digest = bytes.fromhex(ref.sha256)
    except ValueError:
        raise WorkspaceViolationError("Expected a hexadecimal SHA-256 digest") from None
    if len(expected_digest) != 32:
        raise WorkspaceViolationError("Expected a 32-byte SHA-256 digest")
    with registry._open_for_read(ref.root, ref.path, principal, agent_name) as (file, _):
        size, digest = _hash_contents(file, destination=destination, size_limit=ref.size_bytes)
    if size != ref.size_bytes:
        raise WorkspaceViolationError("Workspace artifact size mismatch")
    if digest != expected_digest:
        raise WorkspaceViolationError("Workspace artifact SHA-256 mismatch")


def verify_workspace_ref(
    ref: WorkspaceRef,
    registry: WorkspaceRegistry,
    *,
    principal: str = "local",
    agent_name: str = "",
) -> WorkspaceRef:
    """Verify a reference by hashing the opened workspace file in place."""
    _verify_contents(ref, registry, principal=principal, agent_name=agent_name)
    return ref


def verify_workspace_artifact_in_place(
    artifact: Artifact,
    registry: WorkspaceRegistry,
    *,
    principal: str = "local",
    agent_name: str = "",
) -> WorkspaceRef:
    """Extract and verify a workspace reference without copying its contents."""
    return verify_workspace_ref(
        _extract_workspace_ref(artifact), registry, principal=principal, agent_name=agent_name
    )


def verify_workspace_artifact(
    artifact: Artifact,
    registry: WorkspaceRegistry,
    *,
    principal: str = "local",
    agent_name: str = "",
    max_bytes: int | None = None,
) -> tuple[WorkspaceRef, bytes]:
    """Return the reference and the exact bytes that passed size and digest checks."""
    ref = _extract_workspace_ref(artifact)
    if max_bytes is not None and ref.size_bytes > max_bytes:
        raise WorkspaceViolationError("Workspace artifact exceeds the byte limit")
    with io.BytesIO() as contents:
        _verify_contents(
            ref, registry, principal=principal, agent_name=agent_name, destination=contents
        )
        return ref, contents.getvalue()


@contextmanager
def open_verified_workspace_ref(
    ref: WorkspaceRef,
    registry: WorkspaceRegistry,
    *,
    principal: str = "local",
    agent_name: str = "",
) -> Iterator[tuple[WorkspaceRef, BinaryIO]]:
    """Yield a seekable snapshot of verified bytes with bounded memory use."""
    with TemporaryFile(mode="w+b") as snapshot:
        _verify_contents(
            ref, registry, principal=principal, agent_name=agent_name, destination=snapshot
        )
        snapshot.seek(0)
        yield ref, snapshot
