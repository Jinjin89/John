"""Check the published configuration, pipeline, and links without live accounts."""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pytest

from john import AgentProfile, Capabilities, OutputSpec, coordinator
from john.types import (
    ApprovalPolicy,
    EventSink,
    TurnFinished,
    TurnRequest,
    TurnResult,
    WorkspaceAccess,
)

_ROOT = Path(__file__).resolve().parents[1]
_DOCS = _ROOT / "docs"
_QUICKSTART = _DOCS / "quickstart-claude-codex.md"
_AUTH = _DOCS / "auth-and-fallback.md"
_SOURCE_NOTES = "- CSV exports now keep empty columns in their original order.\n"
_RELEASE_NOTES = f"# Release notes\n\n{_SOURCE_NOTES}"
_VERDICT = {"approved": True, "summary": "The export change is covered.", "issues": []}


def _examples(guide: Path, language: str) -> list[str]:
    pattern = rf"^```{language}[ \t]*\n(.*?)^```[ \t]*$"
    blocks = re.findall(pattern, guide.read_text(encoding="utf-8"), flags=re.MULTILINE | re.S)
    assert blocks, f"No {language} examples in {guide.name}"
    return blocks


@pytest.mark.parametrize(
    "guide, fallback_agents",
    [(_QUICKSTART, ()), (_AUTH, ("claude-gateway", "codex"))],
    ids=["quickstart", "fallback"],
)
def test_documented_configs_parse(
    guide: Path, fallback_agents: tuple[str, ...], tmp_path: Path
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config = tmp_path / "agents.toml"
    for example in _examples(guide, "toml"):
        config.write_text(example, encoding="utf-8")
        profiles, workspaces, ledger = coordinator._read_config(config)
        assert ledger == tmp_path / ".john" / "ledger.db"
        assert workspaces.authorize("demo", "read_write").host_path == workspace
        agents = {profile.name: profile for profile in profiles}
        assert agents.keys() == {"claude", "codex", *fallback_agents}
        assert agents["codex"].options["preflight_auth"] is True
        assert agents["codex"].profile_dir is None
        assert agents["claude"].fallback_agents == fallback_agents
        assert agents["claude"].unset_env == (
            "ANTHROPIC_API_KEY",
            "ANTHROPIC_AUTH_TOKEN",
            "ANTHROPIC_BASE_URL",
        )
        assert agents["claude"].capabilities.enforced_sandbox == {"read_only"}


type _RunTurn = Callable[[Path, WorkspaceAccess, TurnRequest], TurnFinished]


@dataclass(slots=True)
class _MockSession:
    workspace: Path
    access: WorkspaceAccess
    capabilities: Capabilities
    respond: _RunTurn
    native_ref: str | None = None
    ignored_fields: dict[str, str] = field(default_factory=dict)

    async def run_turn(
        self, request: TurnRequest, on_event: EventSink, on_approval: ApprovalPolicy
    ) -> TurnFinished:
        return self.respond(self.workspace, self.access, request)

    async def cancel(self, run_id: str) -> None:
        pass


@dataclass(slots=True)
class _MockDriver:
    respond: _RunTurn

    @asynccontextmanager
    async def open(
        self,
        profile: AgentProfile,
        *,
        workspace_path: Path | None,
        access: WorkspaceAccess,
        resume_ref: str | None = None,
    ) -> AsyncIterator[_MockSession]:
        assert workspace_path is not None
        assert resume_ref is None
        yield _MockSession(workspace_path, access, profile.capabilities, self.respond)


def _draft(workspace: Path, access: WorkspaceAccess, request: TurnRequest) -> TurnFinished:
    assert access == "read_write"
    assert request.outputs == (
        OutputSpec(name="release_notes", path="out/release-notes.md", mediaType="text/markdown"),
    )
    assert "CHANGELOG_NOTES.txt" in request.prompt
    source = (workspace / "CHANGELOG_NOTES.txt").read_text(encoding="utf-8")
    (workspace / request.outputs[0].path).write_text(
        f"# Release notes\n\n{source}", encoding="utf-8"
    )
    return TurnFinished(result=TurnResult(run_id=request.run_id, status="completed"))


def _review(workspace: Path, access: WorkspaceAccess, request: TurnRequest) -> TurnFinished:
    assert access == "read_only"
    assert not request.outputs
    (artifact,) = request.input_refs
    assert (artifact.root, artifact.path) == ("demo", "out/release-notes.md")
    assert artifact.media_type == "text/markdown"
    notes = (workspace / artifact.path).read_bytes()
    assert notes.decode("utf-8") == _RELEASE_NOTES
    assert (artifact.size_bytes, artifact.sha256) == (len(notes), sha256(notes).hexdigest())
    assert request.schema_def is not None
    assert set(request.schema_def["required"]) == _VERDICT.keys()
    assert request.schema_def["additionalProperties"] is False
    return TurnFinished(
        result=TurnResult(run_id=request.run_id, status="completed", text=json.dumps(_VERDICT))
    )


async def test_documented_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    (config,) = _examples(_QUICKSTART, "toml")
    (pipeline,) = _examples(_QUICKSTART, "python")
    (tmp_path / "agents.toml").write_text(config, encoding="utf-8")
    workspace = tmp_path / "workspace"
    (workspace / "out").mkdir(parents=True)
    (workspace / "CHANGELOG_NOTES.txt").write_text(_SOURCE_NOTES, encoding="utf-8")
    drivers = {"codex": _MockDriver(_draft), "claude": _MockDriver(_review)}
    monkeypatch.setattr(coordinator, "_default_drivers", lambda: drivers)
    monkeypatch.chdir(tmp_path)

    namespace = {"__name__": "pipeline"}
    exec(compile(pipeline, str(_QUICKSTART), "exec"), namespace)
    await namespace["main"]()

    output = json.loads(capsys.readouterr().out)
    assert output["verdict"] == _VERDICT
    draft, review = output["lineage"]
    assert (draft["agent_name"], review["agent_name"]) == ("codex", "claude")
    assert draft["parent_run_id"] is None
    assert review["parent_run_id"] == draft["run_id"]


def test_documentation_relative_links_exist() -> None:
    for page in [_ROOT / "README.md", *sorted(_DOCS.rglob("*.md"))]:
        source = page.read_text(encoding="utf-8")
        source = re.sub(
            r"^ {0,3}(`{3,}|~{3,})[^\n]*\n.*?^ {0,3}\1[ \t]*$",
            "",
            source,
            flags=re.MULTILINE | re.DOTALL,
        )
        destinations = re.findall(
            r"(?:\[[^\]\n]*\]\([ \t]*|^ {0,3}\[[^\]\n]+\]:[ \t]*)"
            r"(?:<([^>\n]+)>|([^\s)]+))",
            source,
            flags=re.MULTILINE,
        )
        for angle_bracketed, plain in destinations:
            link = angle_bracketed or plain
            url = urlsplit(link)
            if url.scheme or url.netloc or not url.path or url.path.startswith("/"):
                continue
            assert (page.parent / unquote(url.path)).exists(), (
                f"{page.name} links to missing {link}"
            )
