import errno
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from a2a.helpers.proto_helpers import new_data_part
from a2a.types import Artifact, Message, Part
from google.protobuf.json_format import MessageToDict, MessageToJson, Parse

import john.workspace as workspace_module
from john.types import (
    WORKSPACE_EXT_URI,
    WORKSPACE_REF_MEDIA_TYPE,
    Capabilities,
    OutputSpec,
    UnsupportedError,
    WorkspaceAccess,
    WorkspaceGrant,
    WorkspaceRequest,
    WorkspaceViolationError,
)
from john.workspace import (
    ExtensionRequiredError,
    WorkspaceRegistry,
    assert_outputs_absent,
    build_workspace_artifact,
    open_safe_regular_file,
    open_verified_workspace_ref,
    parse_workspace_request,
    validate_workspace_request,
    verify_workspace_artifact,
    verify_workspace_artifact_in_place,
    verify_workspace_ref,
)


@pytest.fixture
def workspace(tmp_path: Path) -> tuple[WorkspaceRegistry, Path]:
    root = tmp_path / "workspace"
    root.mkdir()
    registry = WorkspaceRegistry()
    registry.register(WorkspaceGrant(root_id="project", host_path=root, max_access="read_write"))
    return registry, root


def test_parse_activated_workspace_metadata_from_message() -> None:
    message = Message(
        metadata={
            WORKSPACE_EXT_URI: {
                "root": "project",
                "access": "read_write",
                "outputs": [
                    {"name": "report", "path": "out/report.json", "mediaType": "application/json"}
                ],
            }
        }
    )
    request = parse_workspace_request(
        MessageToDict(message.metadata), {WORKSPACE_EXT_URI}, required=True
    )
    assert request == WorkspaceRequest(
        root="project",
        access="read_write",
        outputs=(OutputSpec(name="report", path="out/report.json", media_type="application/json"),),
    )


def test_optional_workspace_can_be_omitted() -> None:
    assert parse_workspace_request(None, set(), required=False) is None
    assert parse_workspace_request({"unrelated": {}}, {"other-extension"}, required=False) is None


@pytest.mark.parametrize("metadata", [None, {WORKSPACE_EXT_URI: {"root": "project"}}])
def test_missing_required_activation(metadata: dict | None) -> None:
    with pytest.raises(ExtensionRequiredError):
        parse_workspace_request(metadata, set(), required=True)


def test_supplied_metadata_requires_activation() -> None:
    with pytest.raises(WorkspaceViolationError, match="Activate the workspace extension"):
        parse_workspace_request({WORKSPACE_EXT_URI: {"root": "project"}}, set(), required=False)


@pytest.mark.parametrize(
    "metadata",
    [
        None,
        {},
        {WORKSPACE_EXT_URI: None},
        {WORKSPACE_EXT_URI: ["project"]},
        {WORKSPACE_EXT_URI: {"access": "read_only"}},
        {WORKSPACE_EXT_URI: {"root": "project", "access": "admin"}},
        {WORKSPACE_EXT_URI: {"root": "project", "outputs": {"name": "report", "path": "file"}}},
    ],
)
def test_activated_extension_rejects_malformed_metadata(metadata: dict | None) -> None:
    with pytest.raises(WorkspaceViolationError):
        parse_workspace_request(metadata, {WORKSPACE_EXT_URI}, required=False)


def test_duplicate_output_names_rejected() -> None:
    metadata = {
        WORKSPACE_EXT_URI: {
            "root": "project",
            "access": "read_write",
            "outputs": [{"name": "report", "path": "a"}, {"name": "report", "path": "b"}],
        }
    }
    with pytest.raises(WorkspaceViolationError, match="unique"):
        parse_workspace_request(metadata, {WORKSPACE_EXT_URI}, required=True)


def test_registry_enforces_principal_agent_and_max_access(tmp_path: Path) -> None:
    registry = WorkspaceRegistry()
    grant = WorkspaceGrant(
        root_id="project",
        host_path=tmp_path,
        allowed_principals={"alice"},
        allowed_agents={"qc"},
    )
    registry.register(grant)
    assert registry.get_grant("project") == grant
    assert registry.authorize("project", "read_only", principal="alice", agent_name="qc") == grant
    with pytest.raises(WorkspaceViolationError, match="Unknown workspace root"):
        registry.authorize("unknown", "read_only", principal="alice", agent_name="qc")
    with pytest.raises(WorkspaceViolationError, match="Principal"):
        registry.authorize("project", "read_only", principal="bob", agent_name="qc")
    with pytest.raises(WorkspaceViolationError, match="Agent"):
        registry.authorize("project", "read_only", principal="alice", agent_name="other")
    with pytest.raises(WorkspaceViolationError, match="read-only"):
        registry.authorize("project", "read_write", principal="alice", agent_name="qc")


@pytest.mark.parametrize("kind", ["missing", "file", "symlink"])
def test_registry_requires_a_real_root_directory(tmp_path: Path, kind: str) -> None:
    root = tmp_path / "root"
    if kind == "file":
        root.touch()
    if kind == "symlink":
        root.symlink_to(tmp_path, target_is_directory=True)
    registry = WorkspaceRegistry()
    registry.register(WorkspaceGrant(root_id="project", host_path=root))
    with pytest.raises(WorkspaceViolationError, match="real workspace directory"):
        registry.authorize("project", "read_only")


def test_registry_anchors_relative_root_at_registration(
    workspace: tuple[WorkspaceRegistry, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    registry, root = workspace
    monkeypatch.chdir(root.parent)
    registry.register(WorkspaceGrant(root_id="relative", host_path=Path(root.name)))
    monkeypatch.chdir(root)
    assert registry.authorize("relative", "read_only").host_path == root


@pytest.mark.parametrize("access", ["read_only", "read_write"])
def test_validate_scope_fails_closed_by_default(
    workspace: tuple[WorkspaceRegistry, Path], access: WorkspaceAccess
) -> None:
    registry, _ = workspace
    caps = Capabilities()
    assert caps.enforced_sandbox == frozenset()
    with pytest.raises(UnsupportedError, match="sandbox enforcement"):
        validate_workspace_request(WorkspaceRequest(root="project", access=access), registry, caps)


def test_validate_rejects_unenforceable_access(workspace: tuple[WorkspaceRegistry, Path]) -> None:
    registry, _ = workspace
    read_write = WorkspaceRequest(root="project", access="read_write")
    caps = Capabilities(enforced_sandbox={"read_write"})
    assert validate_workspace_request(read_write, registry, caps) == registry.get_grant("project")
    with pytest.raises(UnsupportedError, match="sandbox enforcement"):
        validate_workspace_request(WorkspaceRequest(root="project"), registry, caps)


def test_validate_requires_write_access_for_outputs(
    workspace: tuple[WorkspaceRegistry, Path],
) -> None:
    registry, _ = workspace
    outputs = (OutputSpec(name="report", path="out/report.txt"),)
    with pytest.raises(WorkspaceViolationError, match="read-write"):
        validate_workspace_request(
            WorkspaceRequest(root="project", outputs=outputs),
            registry,
            Capabilities(enforced_sandbox={"read_only"}),
        )
    request = WorkspaceRequest(root="project", access="read_write", outputs=outputs)
    caps = Capabilities(enforced_sandbox={"read_write"})
    assert validate_workspace_request(request, registry, caps) == registry.get_grant("project")


def test_declared_outputs_can_have_missing_paths(
    workspace: tuple[WorkspaceRegistry, Path],
) -> None:
    registry, root = workspace
    (root / "out").mkdir()
    outputs = (
        OutputSpec(name="report", path="out/report.txt"),
        OutputSpec(name="metadata", path="new/dir/metadata.json"),
    )
    assert_outputs_absent("project", outputs, registry)
    assert not (root / "new").exists()


@pytest.mark.parametrize("kind", ["regular", "directory", "dangling-symlink"])
def test_declared_outputs_reject_any_existing_entry(
    workspace: tuple[WorkspaceRegistry, Path], kind: str
) -> None:
    registry, root = workspace
    (root / "out").mkdir()
    path = root / "out/report.txt"
    if kind == "regular":
        path.write_text("stale")
    elif kind == "directory":
        path.mkdir()
    else:
        path.symlink_to(root / "missing")
    with pytest.raises(WorkspaceViolationError, match="already exists"):
        assert_outputs_absent(
            "project", [OutputSpec(name="report", path="out/report.txt")], registry
        )


def test_declared_outputs_reject_a_symlink_parent(
    workspace: tuple[WorkspaceRegistry, Path],
) -> None:
    registry, root = workspace
    (root / "out").symlink_to(root.parent, target_is_directory=True)
    with pytest.raises(WorkspaceViolationError, match="safely"):
        assert_outputs_absent(
            "project", [OutputSpec(name="report", path="out/report.txt")], registry
        )


@pytest.mark.parametrize(
    "path",
    [
        "",
        "/file",
        "../file",
        "out/../file",
        "./file",
        "file/.",
        "out//file",
        "file/",
        "out/\x00file",
    ],
)
def test_unsafe_paths_rejected_in_metadata_and_at_open(tmp_path: Path, path: str) -> None:
    metadata = {
        WORKSPACE_EXT_URI: {
            "root": "project",
            "access": "read_write",
            "outputs": [{"name": "report", "path": path}],
        }
    }
    with pytest.raises(WorkspaceViolationError):
        parse_workspace_request(metadata, {WORKSPACE_EXT_URI}, required=True)
    with (
        pytest.raises(WorkspaceViolationError, match="workspace-relative POSIX path"),
        open_safe_regular_file(tmp_path, path),
    ):
        pytest.fail("Opened an invalid path")


def test_validate_rejects_unsafe_paths_on_direct_request(
    workspace: tuple[WorkspaceRegistry, Path],
) -> None:
    registry, _ = workspace
    request = WorkspaceRequest(
        root="project", access="read_write", outputs=(OutputSpec(name="report", path="../file"),)
    )
    with pytest.raises(WorkspaceViolationError, match="workspace-relative POSIX path"):
        validate_workspace_request(request, registry, Capabilities(enforced_sandbox={"read_write"}))


@pytest.mark.parametrize("path", ["external/report.txt", "linked-report", "internal-link"])
def test_open_rejects_symlinks(workspace: tuple[WorkspaceRegistry, Path], path: str) -> None:
    _, root = workspace
    outside = root.parent / "outside"
    outside.mkdir()
    report = outside / "report.txt"
    report.touch()
    contained = root / "contained.txt"
    contained.touch()
    (root / "external").symlink_to(outside, target_is_directory=True)
    (root / "linked-report").symlink_to(report)
    (root / "internal-link").symlink_to(contained)
    with pytest.raises(WorkspaceViolationError), open_safe_regular_file(root, path):
        pytest.fail("Followed a symlink")


@pytest.mark.parametrize("path", ["out", "out/missing.txt"])
def test_open_rejects_directory_and_missing_file(
    workspace: tuple[WorkspaceRegistry, Path], path: str
) -> None:
    _, root = workspace
    (root / "out").mkdir()
    with pytest.raises(WorkspaceViolationError), open_safe_regular_file(root, path):
        pytest.fail("Opened an entry with no regular file")


def test_open_closes_directory_descriptors_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "out" / "nested").mkdir(parents=True)
    real_open = os.open
    descriptors = []

    def tracked_open(path: Path | str, flags: int, *, dir_fd: int | None = None) -> int:
        fd = real_open(path, flags, dir_fd=dir_fd)
        descriptors.append(fd)
        return fd

    with monkeypatch.context() as patch:
        patch.setattr(os, "open", tracked_open)
        with (
            pytest.raises(WorkspaceViolationError),
            open_safe_regular_file(tmp_path, "out/nested/missing.txt"),
        ):
            pytest.fail("Opened a missing file")
    assert len(descriptors) >= 3
    for fd in descriptors:
        with pytest.raises(OSError) as exc:
            os.fstat(fd)
        assert exc.value.errno == errno.EBADF


def test_fifo_is_opened_nonblocking_and_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    os.mkfifo(tmp_path / "pipe")
    real_open = os.open

    def checked_open(path: Path | str, flags: int, *, dir_fd: int | None = None) -> int:
        if path == "pipe":
            assert flags & os.O_NONBLOCK
        return real_open(path, flags, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", checked_open)
    with (
        pytest.raises(WorkspaceViolationError, match="regular workspace file"),
        open_safe_regular_file(tmp_path, "pipe"),
    ):
        pytest.fail("Opened a FIFO")


def test_open_reads_pinned_file_after_path_replacement(tmp_path: Path) -> None:
    directory = tmp_path / "out" / "nested"
    directory.mkdir(parents=True)
    path = directory / ".report.txt"
    path.write_bytes(b"original")
    with open_safe_regular_file(tmp_path, "out/nested/.report.txt") as (file, opened_stat):
        path.rename(directory / "previous.txt")
        path.write_bytes(b"replacement")
        pinned_stat = os.fstat(file.fileno())
        assert (opened_stat.st_dev, opened_stat.st_ino) == (pinned_stat.st_dev, pinned_stat.st_ino)
        assert file.read() == b"original"
    assert file.closed


@pytest.mark.parametrize("content", [b"", bytes(range(256)) * 8193], ids=["empty", "multi-chunk"])
def test_artifact_protobuf_round_trip(
    workspace: tuple[WorkspaceRegistry, Path], content: bytes
) -> None:
    registry, root = workspace
    (root / "report.json").write_bytes(content)
    spec = OutputSpec(name="report", path="report.json", media_type="application/json")
    artifact = build_workspace_artifact("project", spec, registry)
    binary = Artifact.FromString(artifact.SerializeToString())
    decoded = Parse(MessageToJson(binary), Artifact())
    assert decoded == artifact
    assert decoded.artifact_id == decoded.name == "report"
    assert list(decoded.extensions) == [WORKSPACE_EXT_URI]
    (part,) = decoded.parts
    assert part.media_type == WORKSPACE_REF_MEDIA_TYPE
    assert part.WhichOneof("content") == "data"
    expected = {
        "root": "project",
        "path": "report.json",
        "mediaType": "application/json",
        "sizeBytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }
    assert MessageToDict(part.data) == expected
    ref, verified = verify_workspace_artifact(decoded, registry)
    assert ref.model_dump(by_alias=True) == expected
    assert verified == content


@pytest.mark.parametrize(
    ("replacement", "error"),
    [
        (b"modified", "SHA-256 mismatch"),
        (b"more bytes", "size mismatch"),
        (b"less", "size mismatch"),
    ],
)
def test_verification_rejects_mutated_file(
    workspace: tuple[WorkspaceRegistry, Path], replacement: bytes, error: str
) -> None:
    registry, root = workspace
    path = root / "report.txt"
    path.write_bytes(b"original")
    artifact = build_workspace_artifact(
        "project", OutputSpec(name="report", path=path.name), registry
    )
    path.write_bytes(replacement)
    with pytest.raises(WorkspaceViolationError, match=error):
        verify_workspace_artifact(artifact, registry)


def test_in_place_verification_hashes_without_staging(
    workspace: tuple[WorkspaceRegistry, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    registry, root = workspace
    path = root / "report.txt"
    path.write_bytes(b"original")
    artifact = build_workspace_artifact(
        "project", OutputSpec(name="report", path=path.name), registry
    )
    ref, _ = verify_workspace_artifact(artifact, registry)

    def unexpected_buffer(*args: object, **kwargs: object) -> None:
        pytest.fail("Verification allocated a content buffer")

    monkeypatch.setattr(workspace_module, "TemporaryFile", unexpected_buffer)
    monkeypatch.setattr(workspace_module, "io", SimpleNamespace(BytesIO=unexpected_buffer))
    assert verify_workspace_ref(ref, registry) == ref
    assert verify_workspace_artifact_in_place(artifact, registry) == ref
    path.write_bytes(b"modified")
    with pytest.raises(WorkspaceViolationError, match="SHA-256 mismatch"):
        verify_workspace_ref(ref, registry)


def test_verify_artifact_enforces_byte_limit_before_opening(
    workspace: tuple[WorkspaceRegistry, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    registry, root = workspace
    path = root / "report.txt"
    content = b"report"
    path.write_bytes(content)
    artifact = build_workspace_artifact(
        "project", OutputSpec(name="report", path=path.name), registry
    )
    _, verified = verify_workspace_artifact(artifact, registry, max_bytes=len(content))
    assert verified == content

    def unexpected_open(*args: object, **kwargs: object) -> None:
        pytest.fail("Verification opened a file above the byte limit")

    monkeypatch.setattr(workspace_module, "open_safe_regular_file", unexpected_open)
    with pytest.raises(WorkspaceViolationError, match="byte limit"):
        verify_workspace_artifact(artifact, registry, max_bytes=len(content) - 1)


def test_verified_handle_keeps_exact_bytes_after_in_place_change(
    workspace: tuple[WorkspaceRegistry, Path],
) -> None:
    registry, root = workspace
    path = root / "report.txt"
    path.write_bytes(b"original")
    artifact = build_workspace_artifact(
        "project", OutputSpec(name="report", path=path.name), registry
    )
    ref, _ = verify_workspace_artifact(artifact, registry)
    with open_verified_workspace_ref(ref, registry) as (checked_ref, file):
        path.write_bytes(b"modified")
        assert checked_ref == ref
        assert os.fstat(file.fileno()).st_size == ref.size_bytes
        assert file.read() == b"original"
    assert file.closed
    with (
        pytest.raises(WorkspaceViolationError, match="SHA-256 mismatch"),
        open_verified_workspace_ref(ref, registry),
    ):
        pytest.fail("Opened an unverified snapshot")


def test_artifact_apis_enforce_read_grants(workspace: tuple[WorkspaceRegistry, Path]) -> None:
    registry, root = workspace
    registry.register(WorkspaceGrant(root_id="restricted", host_path=root, allowed_agents={"qc"}))
    (root / "report.txt").write_bytes(b"report")
    spec = OutputSpec(name="report", path="report.txt")
    with pytest.raises(WorkspaceViolationError, match="Agent"):
        build_workspace_artifact("restricted", spec, registry, agent_name="other")
    artifact = build_workspace_artifact("restricted", spec, registry, agent_name="qc")
    with pytest.raises(WorkspaceViolationError, match="Agent"):
        verify_workspace_artifact(artifact, registry, agent_name="other")
    ref, content = verify_workspace_artifact(artifact, registry, agent_name="qc")
    assert ref.root == "restricted"
    assert content == b"report"


@pytest.mark.parametrize(
    "parts",
    [
        [],
        [Part(url="https://fixture.invalid/reference", media_type=WORKSPACE_REF_MEDIA_TYPE)],
        [new_data_part([], media_type=WORKSPACE_REF_MEDIA_TYPE)],
        [new_data_part({}, media_type=WORKSPACE_REF_MEDIA_TYPE)] * 2,
    ],
)
def test_verify_requires_one_workspace_reference_object(
    workspace: tuple[WorkspaceRegistry, Path], parts: list[Part]
) -> None:
    registry, _ = workspace
    with pytest.raises(WorkspaceViolationError):
        verify_workspace_artifact(Artifact(artifact_id="report", parts=parts), registry)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("sizeBytes", True),
        ("sizeBytes", "1"),
        ("sizeBytes", 1.5),
        ("sizeBytes", float("nan")),
        ("sizeBytes", float("inf")),
        ("sha256", "?" * 64),
        ("path", "../file"),
    ],
)
def test_verify_rejects_malformed_reference_fields(
    workspace: tuple[WorkspaceRegistry, Path], field: str, value: str | bool | float
) -> None:
    registry, root = workspace
    (root / "file").write_bytes(b"a")
    artifact = build_workspace_artifact("project", OutputSpec(name="report", path="file"), registry)
    ref_data = MessageToDict(artifact.parts[0].data)
    ref_data[field] = value
    artifact.parts[0].CopyFrom(new_data_part(ref_data, media_type=WORKSPACE_REF_MEDIA_TYPE))
    with pytest.raises(WorkspaceViolationError):
        verify_workspace_artifact(artifact, registry)
