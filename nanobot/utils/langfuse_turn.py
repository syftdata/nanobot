"""Ambient Langfuse trace context for a whole agent turn.

The OpenAI-compat provider swaps in ``langfuse.openai``'s auto-tracing client
when ``LANGFUSE_SECRET_KEY`` is set (see providers/openai_compat_provider.py),
which uploads each LLM call as its own anonymous, unthreaded trace. This module
adds the missing structure: one enclosing span per agent turn, with trace-level
session/user/tags applied via ``langfuse.propagate_attributes`` so that

- every LLM call (all iterations, retries, fallbacks) of a turn nests into ONE
  trace named for its trigger (``user_reply``, ``digest_run``, ``cron:<job>``);
- all turns in a Slack channel group into one Langfuse *Session* with
  ``session_id = slack:{org}:{channel}`` — the identical key the Syft webapp
  stamps on its digest MCP traces (``mcp:begin_digest`` … ``mcp:submit_digest``),
  so a digest send and the users' replies to it read as one conversation;
- traces are filterable by ``tags``: ``user_reply`` (a human wrote to the bot),
  ``cron``/``digest`` (scheduled digest runs), and ``org:<org_id>``.

Trace attributes MUST go through ``propagate_attributes`` — in langfuse 4.13+
the per-call ``session_id=``/``user_id=``/``tags=`` kwargs on
``chat.completions.create`` are NOT consumed by the wrapper and leak into the
OpenAI SDK, raising ``TypeError``.

No-op (an env check and nothing else) when Langfuse is not configured, so the
gateway behaves identically for deployments without tracing.
"""

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

# Trace-level input/output are capped so a huge tool-laden turn can't bloat the
# span — the nested generations already carry the full payloads.
_IO_CAP = 4000

# Cron job ids seeded by the Syft fleet for the daily digest (poll_orgs.py uses
# "syft-daily-summary-<channelId>"): lets digest runs carry the "digest" tag.
_DIGEST_JOB_PREFIX = "syft-daily-summary"

# Key used by cron/bound_runner.py for the trigger info it attaches to a
# cron-initiated turn's message metadata.
_CRON_TRIGGER_META = "_cron_trigger"


def _org_id(workspace: Path | None) -> str | None:
    """Fleet layout is .../workspaces/{org_id}/workspace — the org is positional
    (there is no org field in the nanobot config)."""
    try:
        if workspace is not None:
            name = Path(workspace).parent.name
            return name or None
    except Exception:
        pass
    return None


def _last_user_text(messages: list[dict[str, Any]] | None) -> str | None:
    """The most recent user-role text — shown as the TRACE's input in Langfuse
    so a session reads like a chat transcript without opening generations."""
    for m in reversed(messages or []):
        if not (isinstance(m, dict) and m.get("role") == "user"):
            continue
        c = m.get("content")
        if isinstance(c, str) and c.strip():
            return c[:_IO_CAP]
        if isinstance(c, list):
            for part in c:
                if isinstance(part, dict) and isinstance(part.get("text"), str) and part["text"].strip():
                    return part["text"][:_IO_CAP]
    return None


def _channel_of(session_key: str | None, chat_id: str | None) -> str | None:
    """Channel component of a session key.

    ``slack:{channel}`` and thread keys ``slack:{channel}:{thread_ts}`` both
    map to ``{channel}`` — thread replies deliberately collapse into the
    channel-level session so a digest and its threaded replies stay together.
    Cron digest turns carry ``session_key = slack:{customer_channel}`` while
    their ``chat_id`` is the reply *sink* channel, so the session key (not
    chat_id) is authoritative when present.
    """
    if session_key:
        parts = session_key.split(":")
        if len(parts) >= 2 and parts[1]:
            return parts[1]
    return chat_id


@contextmanager
def turn_trace(
    *,
    channel: str,
    chat_id: str | None,
    session_key: str | None,
    metadata: dict[str, Any] | None,
    workspace: Path | None,
    initial_messages: list[dict[str, Any]] | None = None,
) -> Iterator[Any]:
    """Wrap one agent turn in a Langfuse span + propagated trace attributes.

    Yields the Langfuse span (or None when tracing is off) so the caller can
    attach the turn's final output — giving the trace a readable input/output
    pair in the Sessions view instead of "undefined".
    """
    if not os.environ.get("LANGFUSE_SECRET_KEY"):
        yield None
        return
    try:
        from langfuse import get_client, propagate_attributes
    except ImportError:
        # Provider already warns about this combination; stay silent here.
        yield None
        return

    md = metadata or {}
    slack_event = (md.get("slack") or {}).get("event") or {}
    cron = md.get(_CRON_TRIGGER_META) or {}
    org = _org_id(workspace)
    chan = _channel_of(session_key, chat_id)

    is_digest = str(cron.get("job_id") or "").startswith(_DIGEST_JOB_PREFIX)
    slack_user = slack_event.get("user")

    if slack_user:
        name = "user_reply"
    elif is_digest:
        name = "digest_run"
    elif cron:
        name = f"cron:{str(cron.get('job_name') or 'job')[:60]}"
    else:
        name = f"turn:{channel}"

    tags = ["nanobot"]
    if slack_user:
        tags.append("user_reply")
    if cron:
        tags.append("cron")
    if is_digest:
        tags.append("digest")
    if org:
        tags.append(f"org:{org}")

    # Same shape as the webapp's digest MCP traces: slack:{org}:{channel}.
    session_id = ":".join(p for p in (channel, org, chan) if p)
    user_id = str(slack_user or ("cron" if cron else "system"))

    try:
        client = get_client()
        span_cm = client.start_as_current_observation(
            as_type="span",
            name=name,
            input=_last_user_text(initial_messages),
        )
    except Exception:
        # Tracing must never take the agent down.
        yield None
        return

    with span_cm as span:
        with propagate_attributes(
            session_id=session_id,
            user_id=user_id,
            tags=tags,
            metadata={
                "chat_id": str(chat_id or ""),
                "session_key": str(session_key or ""),
                **({"cron_run_id": str(cron.get("run_id") or "")} if cron else {}),
            },
        ):
            yield span


def set_turn_output(span: Any, output: str | None) -> None:
    """Attach the turn's final reply to the span — never raises."""
    if span is None or not output:
        return
    try:
        span.update(output=str(output)[:_IO_CAP])
    except Exception:
        pass
