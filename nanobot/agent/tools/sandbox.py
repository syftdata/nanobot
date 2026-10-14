"""Sandbox backends for shell command execution.

To add a new backend, implement a function with the signature:
    _wrap_<name>(command: str, workspace: str, cwd: str) -> str
and register it in _BACKENDS below.
"""

import shlex
from pathlib import Path

from nanobot.config.paths import get_media_dir


def _bwrap(
    command: str,
    workspace: str,
    cwd: str,
    *,
    ro_binds: list[str] | None = None,
    rw_binds: list[str] | None = None,
    unshare_pid: bool = False,
) -> str:
    """Wrap command in a bubblewrap sandbox (requires bwrap in container).

    Only the workspace is bind-mounted read-write; its parent dir (which holds
    config.json) is hidden behind a fresh tmpfs.  The media directory is
    bind-mounted read-only so exec commands can read uploaded attachments.

    ``ro_binds`` / ``rw_binds`` add operator-chosen paths (tool installs, a
    per-org auth dir). They are mounted after the config-dir tmpfs, so a path
    under the workspace parent is re-exposed on purpose and nothing else is.
    ``unshare_pid`` gives the sandbox its own PID namespace so ``/proc`` shows
    only sandboxed processes (no ``/proc/<gateway>/environ``).
    """
    ws = Path(workspace).resolve()
    media = get_media_dir().resolve()

    try:
        sandbox_cwd = str(ws / Path(cwd).resolve().relative_to(ws))
    except ValueError:
        sandbox_cwd = str(ws)

    required = ["/usr"]
    optional = [
        "/bin",
        "/lib",
        "/lib64",
        "/etc/alternatives",
        "/etc/ssl/certs",
        "/etc/resolv.conf",
        "/etc/ld.so.cache",
    ]

    args = ["bwrap", "--new-session", "--die-with-parent", "--setenv", "HOME", str(ws)]
    if unshare_pid:
        args += ["--unshare-pid"]
    for p in required:
        args += ["--ro-bind", p, p]
    for p in optional:
        args += ["--ro-bind-try", p, p]
    args += [
        "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
        "--tmpfs", str(ws.parent),        # mask config dir
        "--dir", str(ws),                 # recreate workspace mount point
        "--bind", str(ws), str(ws),
        "--ro-bind-try", str(media), str(media),  # read-only access to media
    ]
    for p in ro_binds or []:
        args += ["--ro-bind-try", p, p]
    for p in rw_binds or []:
        args += ["--bind-try", p, p]
    args += [
        "--chdir", sandbox_cwd,
        "--", "sh", "-c", command,
    ]
    return shlex.join(args)


_BACKENDS = {"bwrap": _bwrap}


def wrap_command(
    sandbox: str, command: str, workspace: str, cwd: str, **options: object
) -> str:
    """Wrap *command* using the named sandbox backend."""
    if backend := _BACKENDS.get(sandbox):
        return backend(command, workspace, cwd, **options)
    raise ValueError(f"Unknown sandbox backend {sandbox!r}. Available: {list(_BACKENDS)}")
