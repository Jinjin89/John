"""Driver registry for John's headless protocol-family drivers.

All backend-specific identifiers live under `john/drivers/` so core modules stay
100% backend-agnostic.
"""

from __future__ import annotations

from collections.abc import Mapping

from john.drivers.acp import AcpDriver
from john.drivers.claude import ClaudeStreamJsonDriver
from john.drivers.codex import CodexAppServerDriver
from john.drivers.inprocess import (
    InProcessDriver,
    SpecialistHandler,
    make_cutoff_qc_specialist,
    make_echo_specialist,
)
from john.types import Driver


def default_drivers(
    *,
    specialists: Mapping[str, SpecialistHandler] | None = None,
) -> dict[str, Driver]:
    """Return the built-in driver registry keyed by `AgentProfile.kind`."""
    handlers: dict[str, SpecialistHandler] = {
        "invoice-qc": make_cutoff_qc_specialist(),
        "echo": make_echo_specialist(),
    }
    if specialists:
        handlers.update(specialists)
    return {
        "acp": AcpDriver(),
        "claude": ClaudeStreamJsonDriver(),
        "codex": CodexAppServerDriver(),
        "inprocess": InProcessDriver(handlers),
    }


__all__ = [
    "AcpDriver",
    "ClaudeStreamJsonDriver",
    "CodexAppServerDriver",
    "InProcessDriver",
    "default_drivers",
    "make_cutoff_qc_specialist",
    "make_echo_specialist",
]
