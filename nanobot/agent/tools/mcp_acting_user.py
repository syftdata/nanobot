"""Signed acting-user headers for MCP servers that need to know who is chatting.

An MCP server authenticated with an org-level secret (e.g. Syft) cannot tell
which Slack user asked for a tool call, and it must not trust a tool argument
for that. The gateway signs the sender of the inbound Slack event in the HTTP
transport and sends it as request headers:

    x-syft-acting-user: <Slack user id from the Slack event>
    x-syft-acting-ts:   <unix seconds>
    x-syft-acting-sig:  hex HMAC-SHA256(key, f"{org_id}\\n{user}\\n{ts}")

Flow, and why nothing the model controls can change it:

1. The agent loop records ``msg.sender_id`` (Slack ``event["user"]``) of the
   message being processed in a ContextVar, only for the Slack channel.
2. The MCP tool wrapper (our code, not the model) registers that sender under
   a random one-time nonce and puts only the nonce in the call's ``_meta``.
   The MCP SDK sends requests from a separate writer task where the ContextVar
   is not visible, so the nonce is how the request finds its sender.
3. The httpx request hook (the transport) pops the nonce, signs the sender with
   the key it alone holds, and sets the headers. A forged or replayed nonce
   resolves to nothing; caller-supplied x-syft-acting-* headers are stripped.

The key is read from ``keyFile`` at startup, kept in the transport closure,
and the file is registered as protected so file/search/shell tools refuse it.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import time
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Awaitable, Callable

import httpx
from loguru import logger

from nanobot.security.protected_paths import register_protected_path

ACTING_USER_META_KEY = "syftdata.com/actingUserNonce"
HEADER_USER = "x-syft-acting-user"
HEADER_TS = "x-syft-acting-ts"
HEADER_SIG = "x-syft-acting-sig"
_NONCE_TTL_SECONDS = 600

# Channels whose sender_id comes from a verified platform event.
SIGNED_CHANNELS: frozenset[str] = frozenset({"slack"})

_CURRENT_SENDER: ContextVar[str | None] = ContextVar("mcp_acting_sender", default=None)
# nonce -> (sender, created_at). Process-local; never serialized.
_PENDING: dict[str, tuple[str, float]] = {}


def set_acting_sender(channel: str | None, sender_id: str | None) -> None:
    """Record the Slack-event sender of the inbound message being processed."""
    _CURRENT_SENDER.set(str(sender_id) if channel in SIGNED_CHANNELS and sender_id else None)


def current_acting_sender() -> str | None:
    return _CURRENT_SENDER.get()


def sign_acting_user(key: str, org_id: str, user: str, ts: int | str) -> str:
    return hmac.new(key.encode(), f"{org_id}\n{user}\n{ts}".encode(), hashlib.sha256).hexdigest()


def _prune(now: float) -> None:
    for nonce, (_, created) in list(_PENDING.items()):
        if now - created > _NONCE_TTL_SECONDS:
            _PENDING.pop(nonce, None)


def acting_user_call_meta() -> dict[str, Any] | None:
    """``_meta`` for a tools/call: a one-time nonce bound to the current sender."""
    sender = current_acting_sender()
    if not sender:
        return None
    now = time.time()
    _prune(now)
    nonce = secrets.token_urlsafe(24)
    _PENDING[nonce] = (sender, now)
    return {ACTING_USER_META_KEY: nonce}


def load_signing_key(signing: Any) -> str | None:
    """Read the per-org key from ``keyFile`` and protect the file from agent tools."""
    key_file = getattr(signing, "key_file", "") if signing is not None else ""
    if not key_file or not getattr(signing, "org_id", ""):
        return None
    path = Path(key_file).expanduser()
    register_protected_path(path)
    try:
        key = path.read_text().strip()
    except OSError as exc:
        logger.warning("MCP acting-user key file unreadable ({}); signing disabled", type(exc).__name__)
        return None
    return key or None


def make_acting_user_hook(org_id: str, key: str) -> Callable[[httpx.Request], Awaitable[None]]:
    """httpx request hook that signs the nonce-bound Slack sender into headers."""

    async def inject_acting_user_headers(request: httpx.Request) -> None:
        for name in (HEADER_USER, HEADER_TS, HEADER_SIG):
            if name in request.headers:
                del request.headers[name]
        if request.method != "POST":
            return
        try:
            body = json.loads(request.content or b"null")
        except (ValueError, httpx.RequestNotRead):
            return
        if not isinstance(body, dict) or body.get("method") != "tools/call":
            return
        nonce = ((body.get("params") or {}).get("_meta") or {}).get(ACTING_USER_META_KEY)
        if not isinstance(nonce, str):
            return
        entry = _PENDING.pop(nonce, None)
        if entry is None:
            return
        user = entry[0]
        ts = str(int(time.time()))
        request.headers[HEADER_USER] = user
        request.headers[HEADER_TS] = ts
        request.headers[HEADER_SIG] = sign_acting_user(key, org_id, user, ts)

    return inject_acting_user_headers
