"""Progress callback helpers for user-visible output.

These helpers convert agent progress callbacks into outbound chat messages.
Runtime state notifications such as turn lifecycle and model changes live in
``nanobot.bus.runtime_events``.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from nanobot.bus.events import InboundMessage
from nanobot.bus.outbound_events import ProgressEvent, outbound_message_for_event
from nanobot.bus.queue import MessageBus


def build_bus_progress_callback(
    bus: MessageBus,
    msg: InboundMessage,
) -> Callable[..., Awaitable[None]]:
    """Return a callback that publishes progress as outbound messages."""

    async def _publish_progress(
        content: str,
        *,
        tool_hint: bool = False,
        tool_events: list[dict[str, Any]] | None = None,
        file_edit_events: list[dict[str, Any]] | None = None,
        reasoning: bool = False,
        reasoning_end: bool = False,
        tool_hint_label: str | None = None,
    ) -> None:
        metadata = dict(msg.metadata or {})
        if not metadata.get("webui"):
            metadata["_progress"] = True
            metadata["_tool_hint"] = tool_hint
            if tool_hint_label:
                metadata["_tool_hint_label"] = tool_hint_label
            if reasoning:
                metadata["_reasoning_delta"] = True
            if reasoning_end:
                metadata["_reasoning_end"] = True
            if tool_events:
                metadata["_tool_events"] = tool_events
            if file_edit_events:
                metadata["_file_edit_events"] = file_edit_events
        await bus.publish_outbound(
            outbound_message_for_event(
                channel=msg.channel,
                chat_id=msg.chat_id,
                event=ProgressEvent(
                    content=content,
                    tool_hint=tool_hint,
                    tool_hint_label=tool_hint_label,
                    reasoning_delta=reasoning,
                    reasoning_end=reasoning_end,
                    tool_events=tool_events,
                    file_edit_events=file_edit_events,
                ),
                metadata=metadata,
            )
        )

    async def _bus_progress(
        content: str,
        *,
        tool_hint: bool = False,
        tool_events: list[dict[str, Any]] | None = None,
        file_edit_events: list[dict[str, Any]] | None = None,
        reasoning: bool = False,
        reasoning_end: bool = False,
        tool_hint_label: str | None = None,
    ) -> None:
        await _publish_progress(
            content,
            tool_hint=tool_hint,
            tool_events=tool_events,
            file_edit_events=file_edit_events,
            reasoning=reasoning,
            reasoning_end=reasoning_end,
            tool_hint_label=tool_hint_label,
        )

    return _bus_progress
