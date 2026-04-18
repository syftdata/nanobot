"""Tests for the MessageTool / tool_step bridging logic in AgentProgressHook.

These cover the per-tool ``before_run_tool`` / ``after_run_tool`` hook
behaviour that lets the webapp's streaming ``/v1/chat/completions`` endpoint
see tool activity:

  * MessageTool content is streamed via ``on_stream`` (so the webapp renders a
    normal assistant text bubble instead of a silent empty reply).
  * Non-``message`` tool calls emit ``tool_step`` payloads via ``on_tool_step``
    (``running`` when the tool starts, ``completed``/``error`` when it ends) so
    the webapp renders intermediate tool-step cards.
  * Both behaviours are gated on ``channel == "api"`` to avoid duplicating
    output into Slack/Feishu (those channels already receive MessageTool
    content through the bus handler and have no tool-step UI).
"""

from __future__ import annotations

from typing import Any

import pytest

from nanobot.agent.hook import AgentHookContext
from nanobot.agent.progress_hook import AgentProgressHook
from nanobot.providers.base import ToolCallRequest


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
async def test_after_run_tool_streams_message_tool_content_on_api_channel() -> None:
    """When the agent calls MessageTool on the api channel, its ``content``
    argument should be streamed via ``on_stream`` so the webapp sees a normal
    text reply (no bus subscriber exists for channel="api")."""
    deltas: list[str] = []

    async def on_stream(delta: str) -> None:
        deltas.append(delta)

    hook = AgentProgressHook(on_stream=on_stream, channel="api")
    tc = _make_tool_call("message", {"content": "Here you go."})
    ctx = _make_context(tool_calls=[tc], tool_results=["Message sent to api:default"])
    await hook.after_run_tool(ctx, tc, "Message sent to api:default", None)
    assert deltas == ["Here you go."]


@pytest.mark.asyncio
async def test_after_run_tool_strips_think_before_streaming() -> None:
    """strip_think should remove internal <think>...</think> blocks before
    the MessageTool content reaches the client, matching the transformation
    MessageTool.execute applies to its bus-bound copy."""
    deltas: list[str] = []

    async def on_stream(delta: str) -> None:
        deltas.append(delta)

    hook = AgentProgressHook(on_stream=on_stream, channel="api")
    tc = _make_tool_call(
        "message",
        {"content": "<think>secret reasoning</think>\nHello world"},
    )
    ctx = _make_context(tool_calls=[tc], tool_results=["ok"])
    await hook.after_run_tool(ctx, tc, "ok", None)
    joined = "".join(deltas)
    assert "secret reasoning" not in joined
    assert "Hello world" in joined


@pytest.mark.asyncio
async def test_after_run_tool_does_not_stream_message_on_non_api_channel() -> None:
    """For slack/feishu the MessageTool content already reaches the channel
    via its bus subscriber. Streaming it again via on_stream would duplicate
    the user-facing message."""
    deltas: list[str] = []

    async def on_stream(delta: str) -> None:
        deltas.append(delta)

    hook = AgentProgressHook(on_stream=on_stream, channel="slack")
    tc = _make_tool_call("message", {"content": "hi"})
    ctx = _make_context(tool_calls=[tc], tool_results=["ok"])
    await hook.after_run_tool(ctx, tc, "ok", None)
    assert deltas == []


@pytest.mark.asyncio
async def test_tool_step_running_and_completed_for_non_message_tools() -> None:
    """Non-``message`` tool calls on the api channel should produce a
    ``running`` payload when the tool starts and a ``completed`` payload when
    it finishes, carrying the arguments/result for UI rendering."""
    steps: list[dict[str, Any]] = []

    async def on_tool_step(payload: dict[str, Any]) -> None:
        steps.append(payload)

    hook = AgentProgressHook(on_tool_step=on_tool_step, channel="api")
    tc = _make_tool_call("search_leads", {"q": "top"}, call_id="ts_search_1")
    ctx = _make_context(tool_calls=[tc], tool_results=[{"rows": [{"id": 1}, {"id": 2}]}])

    await hook.before_run_tool(ctx, tc)
    await hook.after_run_tool(ctx, tc, {"rows": [{"id": 1}, {"id": 2}]}, None)

    assert [s["status"] for s in steps] == ["running", "completed"]
    completed = steps[1]
    assert completed["id"] == "ts_search_1"
    assert completed["tool"] == "search_leads"
    assert completed["input"] == {"q": "top"}
    assert completed["output"] == {"rows": [{"id": 1}, {"id": 2}]}


@pytest.mark.asyncio
async def test_tool_step_error_status_when_tool_fails() -> None:
    """A fatal tool error should surface as status="error" with an error
    string so the UI can render a failed tool card."""
    steps: list[dict[str, Any]] = []

    async def on_tool_step(payload: dict[str, Any]) -> None:
        steps.append(payload)

    hook = AgentProgressHook(on_tool_step=on_tool_step, channel="api")
    tc = _make_tool_call("search_leads", {"q": "x"}, call_id="ts_err")
    ctx = _make_context(tool_calls=[tc], tool_results=["boom"])

    await hook.after_run_tool(ctx, tc, "boom", RuntimeError("boom"))

    assert steps[-1]["status"] == "error"
    assert "RuntimeError" in steps[-1]["error"]


@pytest.mark.asyncio
async def test_tool_step_skipped_without_callback() -> None:
    """If the caller did not supply an on_tool_step callback (e.g. cli/slack
    runs that don't care about intermediate UI cards), non-message tool
    results should simply be ignored rather than raise."""
    hook = AgentProgressHook(channel="api")
    tc = _make_tool_call("search_leads", {})
    ctx = _make_context(tool_calls=[tc], tool_results=["ok"])
    # Should not raise.
    await hook.before_run_tool(ctx, tc)
    await hook.after_run_tool(ctx, tc, "ok", None)


@pytest.mark.asyncio
async def test_handles_message_and_tool_mix() -> None:
    """A single iteration can contain both a message tool and other tools
    (rare but legal). We should stream the message content AND emit
    tool_step frames for the other tools (never for the message tool)."""
    deltas: list[str] = []
    steps: list[dict[str, Any]] = []

    async def on_stream(delta: str) -> None:
        deltas.append(delta)

    async def on_tool_step(payload: dict[str, Any]) -> None:
        steps.append(payload)

    hook = AgentProgressHook(on_stream=on_stream, on_tool_step=on_tool_step, channel="api")
    search_tc = _make_tool_call("search_leads", {"q": "top"}, call_id="ts_a")
    message_tc = _make_tool_call("message", {"content": "final answer"}, call_id="ts_b")
    ctx = _make_context(
        tool_calls=[search_tc, message_tc],
        tool_results=[{"count": 1}, "Message sent to api:default"],
    )

    await hook.before_run_tool(ctx, search_tc)
    await hook.after_run_tool(ctx, search_tc, {"count": 1}, None)
    await hook.before_run_tool(ctx, message_tc)
    await hook.after_run_tool(ctx, message_tc, "Message sent to api:default", None)

    assert deltas == ["final answer"]
    assert {s["tool"] for s in steps} == {"search_leads"}


@pytest.mark.asyncio
async def test_tool_step_summarizes_non_serializable_result() -> None:
    """Tool results that aren't JSON-serialisable should be coerced to a
    string so the tool_step payload stays JSON-safe on the wire."""
    steps: list[dict[str, Any]] = []

    async def on_tool_step(payload: dict[str, Any]) -> None:
        steps.append(payload)

    class NotJsonable:
        def __repr__(self) -> str:
            return "<NotJsonable>"

    hook = AgentProgressHook(on_tool_step=on_tool_step, channel="api")
    tc = _make_tool_call("weird_tool", {})
    ctx = _make_context(tool_calls=[tc], tool_results=[NotJsonable()])
    await hook.after_run_tool(ctx, tc, NotJsonable(), None)
    assert steps[-1]["output"] == "<NotJsonable>"


def test_summarize_tool_result_strategies() -> None:
    """Smoke-test the static helper used by the tool_step payloads."""
    assert AgentProgressHook._summarize_tool_result(None) is None
    assert AgentProgressHook._summarize_tool_result("hello") == "hello"
    assert AgentProgressHook._summarize_tool_result(42) == 42
    assert AgentProgressHook._summarize_tool_result([1, 2, 3]) == [1, 2, 3]
    assert AgentProgressHook._summarize_tool_result({"k": "v"}) == {"k": "v"}

    class Odd:
        def __repr__(self) -> str:
            return "Odd()"

    assert AgentProgressHook._summarize_tool_result(Odd()) == "Odd()"
