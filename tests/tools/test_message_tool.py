import pytest

from nanobot.agent.tools.message import MessageTool
from nanobot.bus.events import OutboundMessage


@pytest.mark.asyncio
async def test_message_tool_returns_error_when_no_target_context() -> None:
    tool = MessageTool()
    result = await tool.execute(content="test")
    assert result == "Error: No target channel/chat specified"


@pytest.mark.asyncio
async def test_message_tool_records_last_sent_content_on_success() -> None:
    """MessageTool exposes the most recent delivered content so the API
    server can surface it on the non-streaming path when the LLM's own
    final-text slot is empty."""
    sent: list[OutboundMessage] = []

    async def fake_send(msg: OutboundMessage) -> None:
        sent.append(msg)

    tool = MessageTool(
        send_callback=fake_send,
        default_channel="api",
        default_chat_id="default",
    )
    tool.start_turn()
    assert tool.last_sent_content is None

    await tool.execute(content="Here are the results")
    assert len(sent) == 1
    assert tool.last_sent_content == "Here are the results"
    assert tool._sent_in_turn is True


@pytest.mark.asyncio
async def test_message_tool_start_turn_resets_last_sent_content() -> None:
    """Each agent turn must start with a clean slate so stale content from
    the previous turn doesn't leak into the current one's fallback path."""
    async def fake_send(msg: OutboundMessage) -> None:
        pass

    tool = MessageTool(
        send_callback=fake_send,
        default_channel="api",
        default_chat_id="default",
    )
    await tool.execute(content="old content")
    assert tool.last_sent_content == "old content"

    tool.start_turn()
    assert tool.last_sent_content is None
    assert tool._sent_in_turn is False


@pytest.mark.asyncio
async def test_message_tool_does_not_record_content_on_send_failure() -> None:
    """If the bus publish raises, we should not claim the content was
    delivered — both the _sent_in_turn flag and last_sent_content should
    remain unset."""

    async def failing_send(msg: OutboundMessage) -> None:
        raise RuntimeError("bus down")

    tool = MessageTool(
        send_callback=failing_send,
        default_channel="api",
        default_chat_id="default",
    )
    result = await tool.execute(content="never delivered")
    assert result.startswith("Error sending message")
    assert tool.last_sent_content is None
    assert tool._sent_in_turn is False


@pytest.mark.asyncio
async def test_message_tool_cross_chat_send_does_not_set_default_flags() -> None:
    """When MessageTool sends to a chat other than its default context (e.g.
    a spawn/sub-agent directing output elsewhere), we must NOT flag it as
    the turn's user-facing reply and must NOT overwrite last_sent_content —
    otherwise the API path would surface a message the user never asked for."""

    async def fake_send(msg: OutboundMessage) -> None:
        pass

    tool = MessageTool(
        send_callback=fake_send,
        default_channel="api",
        default_chat_id="default",
    )
    await tool.execute(
        content="cross-chat broadcast",
        channel="slack",
        chat_id="C123",
    )
    assert tool.last_sent_content is None
    assert tool._sent_in_turn is False
