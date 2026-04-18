"""Agent hook that adapts runner events into channel progress UI."""

from __future__ import annotations

import inspect
import json
from typing import Any, Awaitable, Callable

from loguru import logger

from nanobot.agent.hook import AgentHook, AgentHookContext
from nanobot.providers.base import ToolCallRequest
from nanobot.utils.helpers import IncrementalThinkExtractor, strip_think
from nanobot.utils.progress_events import (
    build_tool_event_finish_payloads,
    build_tool_event_start_payload,
    invoke_on_progress,
    on_progress_accepts_tool_events,
)
from nanobot.utils.tool_hints import format_tool_hints


class AgentProgressHook(AgentHook):
    """Translate runner lifecycle events into user-visible progress signals."""

    def __init__(
        self,
        on_progress: Callable[..., Awaitable[None]] | None = None,
        on_stream: Callable[[str], Awaitable[None]] | None = None,
        on_stream_end: Callable[..., Awaitable[None]] | None = None,
        on_tool_step: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
        *,
        channel: str = "cli",
        chat_id: str = "direct",
        message_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        session_key: str | None = None,
        tool_hint_max_length: int = 40,
        set_tool_context: Callable[..., None] | None = None,
        on_iteration: Callable[[int], None] | None = None,
    ) -> None:
        super().__init__(reraise=True)
        self._on_progress = on_progress
        self._on_stream = on_stream
        self._on_stream_end = on_stream_end
        self._on_tool_step = on_tool_step
        self._channel = channel
        self._chat_id = chat_id
        self._message_id = message_id
        self._metadata = metadata or {}
        self._session_key = session_key
        self._tool_hint_max_length = tool_hint_max_length
        self._set_tool_context = set_tool_context
        self._on_iteration = on_iteration
        self._stream_buf = ""
        self._think_extractor = IncrementalThinkExtractor()
        self._reasoning_open = False

    def wants_streaming(self) -> bool:
        return self._on_stream is not None

    @staticmethod
    def _strip_think(text: str | None) -> str | None:
        if not text:
            return None
        return strip_think(text) or None

    def _tool_hint(self, tool_calls: list[Any]) -> str:
        return format_tool_hints(tool_calls, max_length=self._tool_hint_max_length)

    @staticmethod
    def _on_progress_accepts(cb: Callable[..., Any], name: str) -> bool:
        try:
            sig = inspect.signature(cb)
        except (TypeError, ValueError):
            return False
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
            return True
        return name in sig.parameters

    async def on_stream(self, context: AgentHookContext, delta: str) -> None:
        prev_clean = strip_think(self._stream_buf)
        self._stream_buf += delta
        new_clean = strip_think(self._stream_buf)
        incremental = new_clean[len(prev_clean) :]

        if await self._think_extractor.feed(self._stream_buf, self.emit_reasoning):
            context.streamed_reasoning = True

        if incremental:
            # Answer text has started; close the reasoning segment so the UI can
            # lock the bubble before the answer renders below it.
            await self.emit_reasoning_end()
            if self._on_stream:
                await self._on_stream(incremental)

    async def on_stream_end(self, context: AgentHookContext, *, resuming: bool) -> None:
        await self.emit_reasoning_end()
        if self._on_stream_end:
            await self._on_stream_end(resuming=resuming)
        self._stream_buf = ""
        self._think_extractor.reset()

    async def before_iteration(self, context: AgentHookContext) -> None:
        if self._on_iteration:
            self._on_iteration(context.iteration)
        logger.debug(
            "Starting agent loop iteration {} for session {}",
            context.iteration,
            self._session_key,
        )

    async def before_execute_tools(self, context: AgentHookContext) -> None:
        if self._on_progress:
            if not self._on_stream and not context.streamed_content:
                thought = self._strip_think(context.response.content if context.response else None)
                if thought:
                    await self._on_progress(thought)
            tool_hint = self._strip_think(self._tool_hint(context.tool_calls))
            tool_events = [build_tool_event_start_payload(tc) for tc in context.tool_calls]
            await invoke_on_progress(
                self._on_progress,
                tool_hint,
                tool_hint=True,
                tool_events=tool_events,
            )
        for tc in context.tool_calls:
            args_str = json.dumps(tc.arguments, ensure_ascii=False)
            logger.info("Tool call: {}({})", tc.name, args_str[:200])
        if self._set_tool_context:
            self._set_tool_context(
                self._channel,
                self._chat_id,
                self._message_id,
                self._metadata,
                session_key=self._session_key,
            )

    async def emit_reasoning(self, reasoning_content: str | None) -> None:
        """Publish a reasoning chunk; channel plugins decide whether to render."""
        if (
            self._on_progress
            and reasoning_content
            and self._on_progress_accepts(self._on_progress, "reasoning")
        ):
            self._reasoning_open = True
            await self._on_progress(reasoning_content, reasoning=True)

    async def emit_reasoning_end(self) -> None:
        """Close the current reasoning stream segment, if any was open."""
        if self._reasoning_open and self._on_progress:
            self._reasoning_open = False
            await self._on_progress("", reasoning_end=True)
        else:
            self._reasoning_open = False

    async def before_run_tool(
        self, context: AgentHookContext, tool_call: ToolCallRequest
    ) -> None:
        """Emit a ``tool_step`` ``running`` frame the moment this tool starts.

        We fire per-tool (rather than once per batch in
        ``before_execute_tools``) so streaming clients can render the "tool
        is working" card immediately, even while earlier tools in the same
        batch are still executing concurrently. The ``message`` tool is
        handled via ``on_stream`` in ``after_run_tool`` instead — it surfaces
        to the user as normal assistant text rather than a tool card.
        """
        if self._channel != "api":
            return
        if self._on_tool_step is None or tool_call.name == "message":
            return
        payload: dict[str, Any] = {
            "id": tool_call.id or f"ts-{tool_call.name}-{id(tool_call)}",
            "tool": tool_call.name,
            "status": "running",
            "input": tool_call.arguments if isinstance(tool_call.arguments, dict) else {},
        }
        try:
            await self._on_tool_step(payload)
        except Exception:
            logger.exception("on_tool_step error (running)")

    async def after_run_tool(
        self,
        context: AgentHookContext,
        tool_call: ToolCallRequest,
        result: Any,
        error: BaseException | None,
    ) -> None:
        """Emit a ``tool_step`` ``completed``/``error`` frame the moment this
        tool finishes, or stream ``message`` tool content as assistant text.

        Gated on the "api" channel so we don't double-deliver into
        slack/feishu (those channels receive ``message`` output via their own
        bus handler).
        """
        if self._channel != "api":
            return

        if tool_call.name == "message":
            if self._on_stream is None:
                return
            raw_content = ""
            if isinstance(tool_call.arguments, dict):
                raw_content = tool_call.arguments.get("content") or ""
            if not isinstance(raw_content, str):
                return
            cleaned = (strip_think(raw_content) or "").strip()
            if cleaned:
                try:
                    await self._on_stream(cleaned)
                except Exception:
                    logger.exception("on_stream error streaming MessageTool content")
            return

        if self._on_tool_step is None:
            return
        payload: dict[str, Any] = {
            "id": tool_call.id or f"ts-{tool_call.name}-{id(tool_call)}",
            "tool": tool_call.name,
            "status": "error" if error is not None else "completed",
            "input": tool_call.arguments if isinstance(tool_call.arguments, dict) else {},
            "output": self._summarize_tool_result(result),
        }
        if error is not None:
            payload["error"] = f"{type(error).__name__}: {error}"[:500]
        try:
            await self._on_tool_step(payload)
        except Exception:
            logger.exception("on_tool_step error (terminal)")

    @staticmethod
    def _summarize_tool_result(result: Any) -> Any:
        """Coerce a tool result into something JSON-serialisable for the
        ``tool_step`` SSE payload: either a primitive/string or a small dict."""
        if result is None:
            return None
        if isinstance(result, (str, int, float, bool)):
            return result
        if isinstance(result, (list, dict)):
            try:
                json.dumps(result)
                return result
            except (TypeError, ValueError):
                return str(result)[:2000]
        return str(result)[:2000]

    async def after_iteration(self, context: AgentHookContext) -> None:
        if (
            self._on_progress
            and context.tool_calls
            and context.tool_events
            and on_progress_accepts_tool_events(self._on_progress)
        ):
            tool_events = build_tool_event_finish_payloads(context)
            if tool_events:
                await invoke_on_progress(
                    self._on_progress,
                    "",
                    tool_hint=False,
                    tool_events=tool_events,
                )
        u = context.usage or {}
        logger.debug(
            "LLM usage: prompt={} completion={} cached={}",
            u.get("prompt_tokens", 0),
            u.get("completion_tokens", 0),
            u.get("cached_tokens", 0),
        )

    def finalize_content(self, context: AgentHookContext, content: str | None) -> str | None:
        return self._strip_think(content)
