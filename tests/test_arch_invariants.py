import ast
import io
import re
import tokenize
from pathlib import Path

_PACKAGE = Path(__file__).resolve().parents[1] / "john"
_NAMES = ("codex", "claude", "acp", "agy", "hermes", "deepagents")
_NAME_PATTERN = re.compile("|".join(_NAMES), re.IGNORECASE)
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def _mentions_backend(text: str) -> bool:
    boundaries = {match.start() for match in _CAMEL_BOUNDARY.finditer(text)}
    for match in _NAME_PATTERN.finditer(text):
        start, end = match.span()
        left = start == 0 or start in boundaries or not text[start - 1].isalnum()
        right = end == len(text) or end in boundaries or not text[end].isalnum()
        if left and right:
            return True
    return False


def test_core_has_no_backend_names() -> None:
    violations = []
    for path in sorted(_PACKAGE.glob("*.py")):
        source = path.read_text(encoding="utf-8")
        tokens = tokenize.generate_tokens(io.StringIO(source).readline)
        for token in tokens:
            if token.type in {tokenize.NAME, tokenize.COMMENT} and _mentions_backend(token.string):
                violations.append(f"{path.name}:{token.start[0]} {token.string!r}")
        for node in ast.walk(ast.parse(source, filename=str(path))):
            if isinstance(node, ast.Constant) and isinstance(node.value, str | bytes):
                value = (
                    node.value
                    if isinstance(node.value, str)
                    else node.value.decode(errors="replace")
                )
                if _mentions_backend(value):
                    violations.append(f"{path.name}:{node.lineno} {value!r}")
    assert not violations, "\n".join(violations)


def test_acp_driver_uses_its_wire_protocol_without_importing_the_sdk() -> None:
    path = _PACKAGE / "drivers" / "acp.py"
    if not path.is_file():
        return
    imports = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=str(path))):
        if isinstance(node, ast.Import):
            imports.update(alias.name.partition(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imports.add(node.module.partition(".")[0])
    assert "acp" not in imports
