import os

import pytest

from nanobot.agent.tools.context import RequestContext, request_context
from nanobot.agent.tools.message import MessageTool
from nanobot.bus.events import OutboundMessage
from nanobot.config.paths import get_workspace_path


@pytest.mark.asyncio
async def test_message_tool_returns_error_when_no_target_context() -> None:
    tool = MessageTool()
    result = await tool.execute(content="test")
    assert result == "Error: No target channel/chat specified"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad",
    [
        "not a list",
        [["ok"], "row-not-a-list"],
        [["ok", 42]],
        [[None]],
    ],
)
async def test_message_tool_rejects_malformed_buttons(bad) -> None:
    """``buttons`` must be ``list[list[str]]``; the tool validates the shape
    up front so a malformed LLM payload errors visibly instead of slipping
    into the channel layer where Telegram would silently reject the frame."""
    tool = MessageTool()
    result = await tool.execute(
        content="hi",
        channel="telegram",
        chat_id="1",
        buttons=bad,
    )
    assert result == "Error: buttons must be a list of list of strings"


@pytest.mark.asyncio
async def test_message_tool_suppresses_delivery_when_active() -> None:
    sent: list[OutboundMessage] = []

    async def _send(msg: OutboundMessage) -> None:
        sent.append(msg)

    tool = MessageTool(send_callback=_send)

    token = tool.set_suppress_delivery(True)
    try:
        result = await tool.execute(content="all clear", channel="telegram", chat_id="1")
    finally:
        tool.reset_suppress_delivery(token)
    assert sent == []
    assert "not delivered" in result

    await tool.execute(content="real", channel="telegram", chat_id="1")
    assert len(sent) == 1
    assert sent[0].content == "real"


@pytest.mark.asyncio
async def test_message_tool_records_media_deliveries() -> None:
    sent: list[OutboundMessage] = []

    async def _send(msg: OutboundMessage) -> None:
        sent.append(msg)

    tool = MessageTool(send_callback=_send)

    await tool.execute(
        content="image",
        channel="websocket",
        chat_id="chat-1",
        media=["/tmp/generated.png"],
    )

    assert sent[0].metadata == {"_record_channel_delivery": True}


@pytest.mark.asyncio
async def test_message_tool_inherits_metadata_for_same_target() -> None:
    sent: list[OutboundMessage] = []

    async def _send(msg: OutboundMessage) -> None:
        sent.append(msg)

    tool = MessageTool(send_callback=_send)
    slack_meta = {"slack": {"thread_ts": "111.222", "channel_type": "channel"}}

    with request_context(RequestContext(channel="slack", chat_id="C123", metadata=slack_meta)):
        await tool.execute(content="thread reply")

    assert sent[0].metadata == slack_meta


@pytest.mark.asyncio
async def test_message_tool_clears_metadata_when_context_has_none() -> None:
    sent: list[OutboundMessage] = []

    async def _send(msg: OutboundMessage) -> None:
        sent.append(msg)

    tool = MessageTool(send_callback=_send)
    rich_context = RequestContext(
        channel="slack",
        chat_id="C123",
        metadata={"slack": {"thread_ts": "111.222", "channel_type": "channel"}},
    )
    with request_context(rich_context):
        await tool.execute(content="thread reply")
    sent.clear()

    with request_context(
        RequestContext(
            channel="slack",
            chat_id="C123",
            metadata={},
        ),
    ):
        await tool.execute(content="plain reply")

    assert sent[0].metadata == {}


@pytest.mark.asyncio
async def test_message_tool_does_not_inherit_metadata_for_cross_target() -> None:
    sent: list[OutboundMessage] = []

    async def _send(msg: OutboundMessage) -> None:
        sent.append(msg)

    tool = MessageTool(send_callback=_send)
    with request_context(
        RequestContext(
            channel="slack",
            chat_id="C123",
            metadata={"slack": {"thread_ts": "111.222", "channel_type": "channel"}},
        ),
    ):
        await tool.execute(content="channel reply", channel="slack", chat_id="C999")

    assert sent[0].metadata == {}


@pytest.mark.asyncio
async def test_message_tool_resolves_relative_media_paths() -> None:
    sent: list[OutboundMessage] = []

    async def _send(msg: OutboundMessage) -> None:
        sent.append(msg)

    tool = MessageTool(send_callback=_send)

    await tool.execute(
        content="see attached",
        channel="telegram",
        chat_id="1",
        media=["output/image.png"],
    )

    expected = str(get_workspace_path() / "output/image.png")
    assert sent[0].media == [expected]


@pytest.mark.asyncio
async def test_message_tool_resolves_relative_media_paths_from_active_workspace(tmp_path) -> None:
    sent: list[OutboundMessage] = []

    async def _send(msg: OutboundMessage) -> None:
        sent.append(msg)

    workspace = tmp_path / "workspace"
    tool = MessageTool(send_callback=_send, workspace=workspace)

    await tool.execute(
        content="see attached",
        channel="telegram",
        chat_id="1",
        media=["output/image.png"],
    )

    assert sent[0].media == [str(workspace / "output/image.png")]


@pytest.mark.asyncio
async def test_message_tool_rejects_outside_workspace_absolute_media_when_restricted(
    tmp_path,
) -> None:
    sent: list[OutboundMessage] = []

    async def _send(msg: OutboundMessage) -> None:
        sent.append(msg)

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "secret.txt"
    outside.write_text("secret", encoding="utf-8")
    tool = MessageTool(send_callback=_send, workspace=workspace, restrict_to_workspace=True)

    result = await tool.execute(
        content="see attached",
        channel="telegram",
        chat_id="1",
        media=[str(outside)],
    )

    assert result.startswith("Error: media path is not allowed:")
    assert "outside allowed directory" in result
    assert sent == []


@pytest.mark.asyncio
async def test_message_tool_allows_workspace_absolute_media_when_restricted(tmp_path) -> None:
    sent: list[OutboundMessage] = []

    async def _send(msg: OutboundMessage) -> None:
        sent.append(msg)

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    image = workspace / "image.png"
    image.write_text("image", encoding="utf-8")
    tool = MessageTool(send_callback=_send, workspace=workspace, restrict_to_workspace=True)

    result = await tool.execute(
        content="see attached",
        channel="telegram",
        chat_id="1",
        media=[str(image)],
    )

    assert result == "Message sent to telegram:1 with 1 attachments"
    assert sent[0].media == [str(image.resolve())]


@pytest.mark.asyncio
async def test_message_tool_passes_through_absolute_media_paths() -> None:
    sent: list[OutboundMessage] = []

    async def _send(msg: OutboundMessage) -> None:
        sent.append(msg)

    tool = MessageTool(send_callback=_send)

    abs_path = os.path.abspath(os.path.join(os.sep, "tmp", "abs_image.png"))

    await tool.execute(
        content="see attached",
        channel="telegram",
        chat_id="1",
        media=[abs_path],
    )

    assert sent[0].media == [abs_path]


@pytest.mark.asyncio
async def test_message_tool_passes_through_url_media_paths() -> None:
    sent: list[OutboundMessage] = []

    async def _send(msg: OutboundMessage) -> None:
        sent.append(msg)

    tool = MessageTool(send_callback=_send)

    url = "https://example.com/image.png"

    await tool.execute(
        content="see attached",
        channel="telegram",
        chat_id="1",
        media=[url],
    )

    assert sent[0].media == [url]


@pytest.mark.asyncio
async def test_message_tool_resolves_mixed_media_paths() -> None:
    sent: list[OutboundMessage] = []

    async def _send(msg: OutboundMessage) -> None:
        sent.append(msg)

    tool = MessageTool(send_callback=_send)

    abs_path = os.path.abspath(os.path.join(os.sep, "tmp", "absolute.png"))

    await tool.execute(
        content="see attached",
        channel="telegram",
        chat_id="1",
        media=[
            "output/relative.png",
            abs_path,
            "https://example.com/url.png",
            "http://example.com/http.png",
        ],
    )

    expected_relative = str(get_workspace_path() / "output/relative.png")
    assert sent[0].media == [
        expected_relative,
        abs_path,
        "https://example.com/url.png",
        "http://example.com/http.png",
    ]


@pytest.mark.asyncio
async def test_message_tool_rejects_wrong_explicit_ws_chat_id(tmp_path) -> None:
    sent: list[OutboundMessage] = []

    async def _send(msg: OutboundMessage) -> None:
        sent.append(msg)

    tool = MessageTool(send_callback=_send)
    conv = "550e8400-e29b-41d4-a716-446655440000"
    f = tmp_path / "doc.md"
    f.write_text("hello", encoding="utf-8")
    with request_context(RequestContext(channel="websocket", chat_id=conv, metadata={})):
        result = await tool.execute(
            content="see file",
            channel="websocket",
            chat_id="anon-deadbeefcafe",
            media=[str(f)],
        )
    assert result.startswith("Error: chat_id does not match")
    assert sent == []


@pytest.mark.asyncio
async def test_message_tool_allows_ws_explicit_when_matches_context(tmp_path) -> None:
    sent: list[OutboundMessage] = []

    async def _send(msg: OutboundMessage) -> None:
        sent.append(msg)

    tool = MessageTool(send_callback=_send)
    conv = "550e8400-e29b-41d4-a716-446655440000"
    f = tmp_path / "doc.md"
    f.write_text("hello", encoding="utf-8")
    with request_context(RequestContext(channel="websocket", chat_id=conv, metadata={})):
        result = await tool.execute(
            content="see file",
            channel="websocket",
            chat_id=conv,
            media=[str(f)],
        )
    assert result.startswith("Message sent")
    assert sent[0].chat_id == conv


@pytest.mark.asyncio
async def test_message_tool_cli_context_may_target_other_ws_chat(tmp_path) -> None:
    """Cron / CLI handlers keep non-websocket defaults; explicit websocket + uuid remains valid."""
    sent: list[OutboundMessage] = []

    async def _send(msg: OutboundMessage) -> None:
        sent.append(msg)

    tool = MessageTool(send_callback=_send)
    target = "550e8400-e29b-41d4-a716-446655440000"
    f = tmp_path / "doc.md"
    f.write_text("hello", encoding="utf-8")
    with request_context(RequestContext(channel="cli", chat_id="direct", metadata={})):
        result = await tool.execute(
            content="ping",
            channel="websocket",
            chat_id=target,
            media=[str(f)],
        )
    assert result.startswith("Message sent")
    assert sent[0].channel == "websocket"
    assert sent[0].chat_id == target


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
