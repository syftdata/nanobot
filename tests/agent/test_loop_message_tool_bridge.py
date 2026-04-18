"""Tests for the MessageTool / tool_step bridging logic added to _LoopHook.

These cover the ``after_execute_tools`` hook behaviour that lets the webapp's
streaming ``/v1/chat/completions`` endpoint see tool activity:

  * MessageTool content is streamed via ``on_stream`` (so the webapp renders a
    normal assistant text bubble instead of a silent empty reply).
  * Non-``message`` tool calls emit ``tool_step`` payloads via ``on_tool_step``
    (so the webapp renders intermediate tool-step cards).
  * Both behaviours are gated on ``channel == "api"`` to avoid duplicating
    output into Slack/Feishu (those channels already receive MessageTool
    content through the bus handler and have no tool-step UI).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from nanobot.agent.hook import AgentHookContext
from nanobot.agent.loop import _LoopHook
from nanobot.providers.base import ToolCallRequest


def _loop_stub() -> MagicMock:
    return MagicMock()


def _make_tool_call(name: str, arguments: dict[str, Any], call_id: str = "tc_1") -> ToolCallRequest:
    return ToolCallRequest(id=call_id, name=name, arguments=arguments)


def _make_context(tool_calls: list[ToolCallRequest], tool_results: list[Any]) -> AgentHookContext:
    return AgentHookContext(
        iteration=0,
        messages=[],
        tool_calls=tool_calls,
        tool_results=tool_results,
    )


@pytest.mark.asyncio
async def test_after_execute_tools_streams_message_tool_content_on_api_channel() -> None:
    """When the agent calls MessageTool on the api channel, its ``content``
    argument should be streamed via ``on_stream`` so the webapp sees a normal
    text reply (no bus subscriber exists for channel="api")."""
    deltas: list[str] = []

    async def on_stream(delta: str) -> None:
        deltas.append(delta)

    hook = _LoopHook(
        _loop_stub(),
        on_stream=on_stream,
        channel="api",
    )
    ctx = _make_context(
        tool_calls=[_make_tool_call("message", {"content": "Here you go."})],
        tool_results=["Message sent to api:default"],
    )
    await hook.after_execute_tools(ctx)
    assert deltas == ["Here you go."]


@pytest.mark.asyncio
async def test_after_execute_tools_strips_think_before_streaming() -> None:
    """strip_think should remove internal <think>...</think> blocks before
    the MessageTool content reaches the client, matching the transformation
    MessageTool.execute applies to its bus-bound copy."""
    deltas: list[str] = []

    async def on_stream(delta: str) -> None:
        deltas.append(delta)

    hook = _LoopHook(_loop_stub(), on_stream=on_stream, channel="api")
    ctx = _make_context(
        tool_calls=[
            _make_tool_call(
                "message",
                {"content": "<think>secret reasoning</think>\nHello world"},
            )
        ],
        tool_results=["ok"],
    )
    await hook.after_execute_tools(ctx)
    joined = "".join(deltas)
    assert "secret reasoning" not in joined
    assert "Hello world" in joined


@pytest.mark.asyncio
async def test_after_execute_tools_does_not_stream_message_on_non_api_channel() -> None:
    """For slack/feishu the MessageTool content already reaches the channel
    via its bus subscriber. Streaming it again via on_stream would duplicate
    the user-facing message."""
    deltas: list[str] = []

    async def on_stream(delta: str) -> None:
        deltas.append(delta)

    hook = _LoopHook(_loop_stub(), on_stream=on_stream, channel="slack")
    ctx = _make_context(
        tool_calls=[_make_tool_call("message", {"content": "hi"})],
        tool_results=["ok"],
    )
    await hook.after_execute_tools(ctx)
    assert deltas == []


@pytest.mark.asyncio
async def test_after_execute_tools_emits_tool_step_for_non_message_tools() -> None:
    """Non-``message`` tool calls on the api channel should produce one
    ``tool_step`` payload per call with status="completed" and the
    arguments/result carried through for UI rendering."""
    steps: list[dict[str, Any]] = []

    async def on_tool_step(payload: dict[str, Any]) -> None:
        steps.append(payload)

    hook = _LoopHook(_loop_stub(), on_tool_step=on_tool_step, channel="api")
    ctx = _make_context(
        tool_calls=[
            _make_tool_call(
                "search_leads",
                {"q": "top"},
                call_id="ts_search_1",
            )
        ],
        tool_results=[{"rows": [{"id": 1}, {"id": 2}]}],
    )
    await hook.after_execute_tools(ctx)
    assert len(steps) == 1
    step = steps[0]
    assert step["id"] == "ts_search_1"
    assert step["tool"] == "search_leads"
    assert step["status"] == "completed"
    assert step["input"] == {"q": "top"}
    assert step["output"] == {"rows": [{"id": 1}, {"id": 2}]}


@pytest.mark.asyncio
async def test_after_execute_tools_skips_tool_step_without_callback() -> None:
    """If the caller did not supply an on_tool_step callback (e.g. cli/slack
    runs that don't care about intermediate UI cards), non-message tool
    results should simply be ignored rather than raise."""
    hook = _LoopHook(_loop_stub(), channel="api")
    ctx = _make_context(
        tool_calls=[_make_tool_call("search_leads", {})],
        tool_results=["ok"],
    )
    # Should not raise.
    await hook.after_execute_tools(ctx)


@pytest.mark.asyncio
async def test_after_execute_tools_handles_message_and_tool_mix() -> None:
    """A single iteration can contain both a message tool and other tools
    (rare but legal). We should stream the message content AND emit
    tool_step frames for the other tools."""
    deltas: list[str] = []
    steps: list[dict[str, Any]] = []

    async def on_stream(delta: str) -> None:
        deltas.append(delta)

    async def on_tool_step(payload: dict[str, Any]) -> None:
        steps.append(payload)

    hook = _LoopHook(
        _loop_stub(),
        on_stream=on_stream,
        on_tool_step=on_tool_step,
        channel="api",
    )
    ctx = _make_context(
        tool_calls=[
            _make_tool_call("search_leads", {"q": "top"}, call_id="ts_a"),
            _make_tool_call("message", {"content": "final answer"}, call_id="ts_b"),
        ],
        tool_results=[{"count": 1}, "Message sent to api:default"],
    )
    await hook.after_execute_tools(ctx)
    assert deltas == ["final answer"]
    assert len(steps) == 1
    assert steps[0]["tool"] == "search_leads"


@pytest.mark.asyncio
async def test_after_execute_tools_summarizes_non_serializable_result() -> None:
    """Tool results that aren't JSON-serialisable should be coerced to a
    string so the tool_step payload stays JSON-safe on the wire."""
    steps: list[dict[str, Any]] = []

    async def on_tool_step(payload: dict[str, Any]) -> None:
        steps.append(payload)

    class NotJsonable:
        def __repr__(self) -> str:
            return "<NotJsonable>"

    hook = _LoopHook(_loop_stub(), on_tool_step=on_tool_step, channel="api")
    ctx = _make_context(
        tool_calls=[_make_tool_call("weird_tool", {})],
        tool_results=[NotJsonable()],
    )
    await hook.after_execute_tools(ctx)
    assert steps[0]["output"] == "<NotJsonable>"


def test_loophook_summarize_tool_result_strategies() -> None:
    """Smoke-test the static helper used by after_execute_tools."""
    from nanobot.agent.loop import _LoopHook

    assert _LoopHook._summarize_tool_result(None) is None
    assert _LoopHook._summarize_tool_result("hello") == "hello"
    assert _LoopHook._summarize_tool_result(42) == 42
    assert _LoopHook._summarize_tool_result([1, 2, 3]) == [1, 2, 3]
    assert _LoopHook._summarize_tool_result({"k": "v"}) == {"k": "v"}

    class Odd:
        def __repr__(self) -> str:
            return "Odd()"

    assert _LoopHook._summarize_tool_result(Odd()) == "Odd()"
