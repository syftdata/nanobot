"""Paths that agent tools (file, search, shell) must never read.

The gateway registers secrets it loads for its own use (e.g. the per-org MCP
acting-user signing key file) so the model cannot exfiltrate them through
read_file / grep / exec. This is a best-effort guard inside one OS user; the
hard boundary is the exec sandbox (``tools.exec.sandbox``) and file ownership.
"""

from __future__ import annotations

import re
from pathlib import Path

_PROTECTED: set[Path] = set()
_SHELL_TOKENS: set[str] = set()

# Reading another process's environment / memory is never needed by the agent.
_ALWAYS_DENIED_SHELL = [
    r"/proc/[^\s]*/(?:environ|mem|maps|cmdline)",
    r"/proc/self\b",
]


def register_protected_path(path: str | Path) -> None:
    resolved = Path(path).expanduser().resolve()
    _PROTECTED.add(resolved)
    _SHELL_TOKENS.add(str(resolved).lower())
    _SHELL_TOKENS.add(resolved.name.lower())
    if resolved.parent != resolved:
        _SHELL_TOKENS.add(str(resolved.parent).lower())


def clear_protected_paths() -> None:
    """Test helper."""
    _PROTECTED.clear()
    _SHELL_TOKENS.clear()


def is_protected(path: str | Path) -> bool:
    """True when `path` is, or is inside, a protected path (symlinks resolved)."""
    if not _PROTECTED:
        return False
    resolved = Path(path).expanduser().resolve()
    return any(resolved == p or p in resolved.parents for p in _PROTECTED)


def contains_protected(path: str | Path) -> bool:
    """True when a recursive walk of `path` would reach a protected path."""
    if not _PROTECTED:
        return False
    resolved = Path(path).expanduser().resolve()
    return is_protected(resolved) or any(resolved in p.parents for p in _PROTECTED)


def shell_command_touches_protected(command: str) -> bool:
    lower = command.lower()
    if any(re.search(p, lower) for p in _ALWAYS_DENIED_SHELL):
        return True
    return any(tok and tok in lower for tok in _SHELL_TOKENS)
