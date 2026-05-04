"""Slack channel implementation using Socket Mode or Webhook."""

from __future__ import annotations

import asyncio
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any

import httpx
from loguru import logger
from pydantic import Field
from slack_sdk.web.async_client import AsyncWebClient
from slackify_markdown import slackify_markdown

from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.base import BaseChannel
from nanobot.config.paths import get_media_dir
from nanobot.config.schema import Base
from nanobot.utils.helpers import safe_filename, split_message


class SlackDMConfig(Base):
    """Slack DM policy configuration."""

    enabled: bool = True
    policy: str = "open"
    allow_from: list[str] = Field(default_factory=list)


class SlackConfig(Base):
    """Slack channel configuration."""

    enabled: bool = False
    mode: str = "socket"
    webhook_path: str = "/slack/events"
    webhook_host: str = "127.0.0.1"
    webhook_port: int = 18800
    webhook_secret: str = ""
    bot_token: str = ""
    channel_tokens: dict[str, str] = Field(default_factory=dict)
    app_token: str = ""
    user_token_read_only: bool = True
    reply_in_thread: bool = True
    react_emoji: str = "eyes"
    done_emoji: str = "white_check_mark"
    typing_status: str = ""
    include_thread_context: bool = True
    thread_context_limit: int = 20
    allow_from: list[str] = Field(default_factory=list)
    group_policy: str = "mention"
    group_allow_from: list[str] = Field(default_factory=list)
    dm: SlackDMConfig = Field(default_factory=SlackDMConfig)


SLACK_MAX_MESSAGE_LEN = 39_000  # Slack API allows ~40k; leave margin
SLACK_DOWNLOAD_TIMEOUT = 30.0
_HTML_DOWNLOAD_PREFIXES = (b"<!doctype html", b"<html")


class SlackChannel(BaseChannel):
    """Slack channel using Socket Mode."""

    name = "slack"
    display_name = "Slack"
    _SLACK_ID_RE = re.compile(r"^[CDGUW][A-Z0-9]{2,}$")
    _SLACK_CHANNEL_REF_RE = re.compile(r"^<#([A-Z0-9]+)(?:\|[^>]+)?>$")
    _SLACK_USER_REF_RE = re.compile(r"^<@([A-Z0-9]+)(?:\|[^>]+)?>$")

    @classmethod
    def default_config(cls) -> dict[str, Any]:
        return SlackConfig().model_dump(by_alias=True)

    _THREAD_CONTEXT_CACHE_LIMIT = 10_000

    def __init__(self, config: Any, bus: MessageBus):
        if isinstance(config, dict):
            config = SlackConfig.model_validate(config)
        super().__init__(config, bus)
        self.config: SlackConfig = config
        self._web_client: AsyncWebClient | None = None
        self._channel_clients: dict[str, AsyncWebClient] = {}
        self._socket_client: Any | None = None
        self._bot_user_id: str | None = None
        self._all_bot_user_ids: set[str] = set()
        self._target_cache: dict[str, str] = {}
        self._webhook_runner: Any | None = None
        self._mention_threads: OrderedDict[str, None] = OrderedDict()
        self._thread_context_attempted: set[str] = set()

    def _get_client(self, channel_id: str | None = None) -> AsyncWebClient | None:
        """Get the appropriate Slack client for the given channel."""
        if not channel_id:
            return self._web_client
        token = self.config.channel_tokens.get(channel_id)
        if not token or token not in self._channel_clients:
            return self._web_client
        return self._channel_clients[token]

    async def start(self) -> None:
        """Start the Slack channel in the configured mode (socket or webhook)."""
        if not self.config.bot_token:
            logger.error("Slack bot_token not configured")
            return

        self._running = True
        self._web_client = AsyncWebClient(token=self.config.bot_token)

        try:
            auth = await self._web_client.auth_test()
            self._bot_user_id = auth.get("user_id")
            if self._bot_user_id:
                self._all_bot_user_ids.add(self._bot_user_id)
            logger.info("Slack bot connected as {}", self._bot_user_id)
        except Exception as e:
            logger.warning("Slack auth_test failed: {}", e)

        for token in set(self.config.channel_tokens.values()):
            if token == self.config.bot_token or token in self._channel_clients:
                continue
            try:
                client = AsyncWebClient(token=token)
                auth = await client.auth_test()
                user_id = auth.get("user_id")
                if user_id:
                    self._all_bot_user_ids.add(user_id)
                logger.info("Other slack bots connected as {}", user_id)
                self._channel_clients[token] = client
            except Exception as e:
                logger.warning("Slack auth_test failed for channel token: {}", e)

        if self.config.mode == "socket":
            await self._start_socket_mode()
        elif self.config.mode == "webhook":
            await self._start_webhook_server()
        else:
            logger.error("Unsupported Slack mode: {}", self.config.mode)

    async def _start_socket_mode(self) -> None:
        """Connect via Socket Mode (original behavior)."""
        if not self.config.app_token:
            logger.error("Slack app_token required for socket mode")
            return

        from slack_sdk.socket_mode.request import SocketModeRequest  # noqa: F811
        from slack_sdk.socket_mode.websockets import SocketModeClient

        self._socket_client = SocketModeClient(
            app_token=self.config.app_token,
            web_client=self._web_client,
        )
        self._socket_client.socket_mode_request_listeners.append(self._on_socket_request)

        logger.info("Starting Slack Socket Mode client...")
        await self._socket_client.connect()

        while self._running:
            await asyncio.sleep(1)

    async def _start_webhook_server(self) -> None:
        """Start an HTTP server that accepts event payloads from the event router."""
        try:
            from aiohttp import web
        except ImportError:
            logger.error(
                "aiohttp is required for webhook mode: pip install nanobot-ai[api]"
            )
            return

        async def handle_event(request: web.Request) -> web.Response:
            if self.config.webhook_secret:
                auth_header = request.headers.get("Authorization", "")
                if auth_header != f"Bearer {self.config.webhook_secret}":
                    return web.Response(status=401, text="Unauthorized")
            try:
                payload = await request.json()
            except Exception:
                return web.Response(status=400, text="Invalid JSON")
            asyncio.create_task(self._process_slack_event(payload))
            return web.Response(status=200, text="ok")

        async def handle_health(_: web.Request) -> web.Response:
            return web.Response(
                text='{"status":"ok"}',
                content_type="application/json",
            )

        app = web.Application()
        app.router.add_post(self.config.webhook_path, handle_event)
        app.router.add_get("/health", handle_health)

        runner = web.AppRunner(app, access_log=None)
        await runner.setup()
        self._webhook_runner = runner
        site = web.TCPSite(runner, self.config.webhook_host, self.config.webhook_port)
        await site.start()
        logger.info(
            "Slack webhook server listening on {}:{}{}",
            self.config.webhook_host,
            self.config.webhook_port,
            self.config.webhook_path,
        )

        while self._running:
            await asyncio.sleep(1)

        await runner.cleanup()

    async def stop(self) -> None:
        """Stop the Slack client."""
        self._running = False
        if self._socket_client:
            try:
                await self._socket_client.close()
            except Exception as e:
                logger.warning("Slack socket close failed: {}", e)
            self._socket_client = None
        if self._webhook_runner:
            try:
                await self._webhook_runner.cleanup()
            except Exception as e:
                logger.warning("Slack webhook runner cleanup failed: {}", e)
            self._webhook_runner = None

    async def send(self, msg: OutboundMessage) -> None:
        """Send a message through Slack."""
        if not self._web_client:
            logger.warning("Slack client not running")
            return
        try:
            target_chat_id = await self._resolve_target_chat_id(msg.chat_id)
            slack_meta = msg.metadata.get("slack", {}) if msg.metadata else {}
            thread_ts = slack_meta.get("thread_ts")
            origin_chat_id = str((slack_meta.get("event", {}) or {}).get("channel") or msg.chat_id)
            client = self._get_client(origin_chat_id)
            if not client:
                logger.error("Slack client not available for chat_id: {}", origin_chat_id)
                return

            # Reply in the same thread the inbound message belongs to (works
            # for both real channel threads and DM threads). When the agent
            # is forwarding to a different channel, drop thread_ts because it
            # only makes sense within the originating conversation.
            thread_ts_param = thread_ts if thread_ts and target_chat_id == origin_chat_id else None

            # Route tool hints to typing status instead of posting as messages
            if (msg.metadata or {}).get("_tool_hint"):
                if self.config.typing_status and thread_ts:
                    status_text = (msg.metadata or {}).get("_tool_hint_label") or msg.content or ""
                    await self._set_typing_status(origin_chat_id, thread_ts, status_text)
                return

            is_progress = (msg.metadata or {}).get("_progress", False)
            if is_progress and not msg.content:
                pass  # skip empty progress messages (e.g. tool-event-only updates)
            elif msg.content or not (msg.media or []):
                mrkdwn = self._to_mrkdwn(msg.content) if msg.content else " "
                buttons = getattr(msg, "buttons", None) or []
                chunks = split_message(mrkdwn, SLACK_MAX_MESSAGE_LEN)
                for index, chunk in enumerate(chunks):
                    kwargs: dict[str, Any] = dict(
                        channel=target_chat_id, text=chunk, thread_ts=thread_ts_param,
                    )
                    if buttons and index == len(chunks) - 1:
                        kwargs["blocks"] = self._build_button_blocks(chunk, buttons)
                    await client.chat_postMessage(**kwargs)

            for media_path in msg.media or []:
                try:
                    await client.files_upload_v2(
                        channel=target_chat_id,
                        file=media_path,
                        thread_ts=thread_ts_param,
                    )
                except Exception as e:
                    logger.error("Failed to upload file {}: {}", media_path, e)

            # Update reaction emoji when the final (non-progress) response is sent
            if not (msg.metadata or {}).get("_progress"):
                event = slack_meta.get("event", {})
                msg_ts = event.get("ts")
                if self.config.typing_status:
                    await self._update_react_emoji(origin_chat_id, thread_ts or msg_ts)
                else:
                    await self._update_react_emoji(origin_chat_id, msg_ts)

        except Exception as e:
            logger.error("Error sending Slack message: {}", e)
            raise

    async def _resolve_target_chat_id(self, target: str) -> str:
        """Resolve human-friendly Slack targets to concrete IDs when needed."""
        if not self._web_client:
            return target

        target = target.strip()
        if not target:
            return target

        if match := self._SLACK_CHANNEL_REF_RE.fullmatch(target):
            return match.group(1)
        if match := self._SLACK_USER_REF_RE.fullmatch(target):
            return await self._open_dm_for_user(match.group(1))
        if self._SLACK_ID_RE.fullmatch(target):
            if target.startswith(("U", "W")):
                return await self._open_dm_for_user(target)
            return target

        if target.startswith("#"):
            return await self._resolve_channel_name(target[1:])
        if target.startswith("@"):
            return await self._resolve_user_handle(target[1:])

        try:
            return await self._resolve_channel_name(target)
        except ValueError:
            return await self._resolve_user_handle(target)

    async def _resolve_channel_name(self, name: str) -> str:
        normalized = self._normalize_target_name(name)
        if not normalized:
            raise ValueError("Slack target channel name is empty")

        cache_key = f"channel:{normalized}"
        if cache_key in self._target_cache:
            return self._target_cache[cache_key]

        client = self._get_client()
        if not client:
            raise ValueError("Slack client not available")

        cursor: str | None = None
        while True:
            response = await client.conversations_list(
                types="public_channel,private_channel",
                exclude_archived=True,
                limit=200,
                cursor=cursor,
            )
            for channel in response.get("channels", []):
                if self._normalize_target_name(str(channel.get("name") or "")) == normalized:
                    channel_id = str(channel.get("id") or "")
                    if channel_id:
                        self._target_cache[cache_key] = channel_id
                        return channel_id
            cursor = ((response.get("response_metadata") or {}).get("next_cursor") or "").strip()
            if not cursor:
                break

        raise ValueError(
            f"Slack channel '{name}' was not found. Use a joined channel name like "
            f"'#general' or a concrete channel ID."
        )

    async def _resolve_user_handle(self, handle: str) -> str:
        normalized = self._normalize_target_name(handle)
        if not normalized:
            raise ValueError("Slack target user handle is empty")

        cache_key = f"user:{normalized}"
        if cache_key in self._target_cache:
            return self._target_cache[cache_key]

        client = self._get_client()
        if not client:
            raise ValueError("Slack client not available")

        cursor: str | None = None
        while True:
            response = await client.users_list(limit=200, cursor=cursor)
            for member in response.get("members", []):
                if self._member_matches_handle(member, normalized):
                    user_id = str(member.get("id") or "")
                    if not user_id:
                        continue
                    dm_id = await self._open_dm_for_user(user_id)
                    self._target_cache[cache_key] = dm_id
                    return dm_id
            cursor = ((response.get("response_metadata") or {}).get("next_cursor") or "").strip()
            if not cursor:
                break

        raise ValueError(
            f"Slack user '{handle}' was not found. Use '@name' or a concrete DM/channel ID."
        )

    async def _open_dm_for_user(self, user_id: str) -> str:
        client = self._get_client()
        if not client:
            raise ValueError("Slack client not available")
        response = await client.conversations_open(users=user_id)
        channel_id = str(((response.get("channel") or {}).get("id")) or "")
        if not channel_id:
            raise ValueError(f"Slack DM target for user '{user_id}' could not be opened.")
        return channel_id

    @staticmethod
    def _normalize_target_name(value: str) -> str:
        return value.strip().lstrip("#@").lower()

    @classmethod
    def _member_matches_handle(cls, member: dict[str, Any], normalized: str) -> bool:
        profile = member.get("profile") or {}
        candidates = {
            str(member.get("name") or ""),
            str(profile.get("display_name") or ""),
            str(profile.get("display_name_normalized") or ""),
            str(profile.get("real_name") or ""),
            str(profile.get("real_name_normalized") or ""),
        }
        return normalized in {cls._normalize_target_name(candidate) for candidate in candidates if candidate}

    async def _on_socket_request(
        self,
        client: Any,
        req: Any,
    ) -> None:
        """Handle incoming Socket Mode requests."""
        if req.type == "interactive":
            await self._on_block_action(client, req)
            return
        if req.type != "events_api":
            return

        from slack_sdk.socket_mode.response import SocketModeResponse

        await client.send_socket_mode_response(
            SocketModeResponse(envelope_id=req.envelope_id)
        )

        await self._process_slack_event(req.payload or {})

    async def _process_slack_event(self, payload: dict) -> None:
        """Process a Slack event payload (shared by socket and webhook modes)."""
        event = payload.get("event") or {}
        event_type = event.get("type")

        if event_type not in ("message", "app_mention"):
            return

        sender_id = event.get("user")
        chat_id = event.get("channel")

        subtype = event.get("subtype")
        if subtype and subtype != "file_share":
            return
        if sender_id in self._all_bot_user_ids:
            return

        text = event.get("text") or ""
        if event_type == "message" and any(f"<@{bot_id}>" in text for bot_id in self._all_bot_user_ids):
            return

        logger.debug(
            "Slack event: type={} subtype={} user={} channel={} channel_type={} text={}",
            event_type,
            subtype,
            sender_id,
            chat_id,
            event.get("channel_type"),
            text[:80],
        )
        if not sender_id or not chat_id:
            return

        channel_type = event.get("channel_type") or ""

        if not self._is_allowed(sender_id, chat_id, channel_type):
            return

        event_thread_ts = event.get("thread_ts")

        if channel_type != "im" and not self._should_respond_in_channel(
            event_type, text, chat_id, thread_ts=event_thread_ts,
        ):
            return

        if event_type == "app_mention":
            thread_key = event_thread_ts or event.get("ts")
            if thread_key:
                self._track_mention_thread(thread_key)

        text = self._strip_bot_mention(text)

        event_ts = event.get("ts")
        raw_thread_ts = event.get("thread_ts")
        thread_ts = raw_thread_ts
        if (
            self.config.reply_in_thread
            and not thread_ts
            and channel_type != "im"
        ):
            thread_ts = event_ts

        if self.config.typing_status:
            await self._set_typing_status(chat_id, thread_ts or event_ts, self.config.typing_status)
        else:
            try:
                client = self._get_client(chat_id)
                if client and event.get("ts"):
                    await client.reactions_add(
                        channel=chat_id,
                        name=self.config.react_emoji,
                        timestamp=event.get("ts"),
                    )
            except Exception as e:
                logger.debug("Slack reactions_add failed: {}", e)

        session_key = (
            f"slack:{chat_id}:{thread_ts}" if thread_ts and raw_thread_ts else None
        )
        media_paths: list[str] = []
        file_markers: list[str] = []
        for file_info in event.get("files") or []:
            if not isinstance(file_info, dict):
                continue
            file_path, marker = await self._download_slack_file(file_info)
            if file_path:
                media_paths.append(file_path)
            if marker:
                file_markers.append(marker)

        is_slash = text.strip().startswith("/")
        content = text if is_slash else await self._with_thread_context(
            text,
            chat_id=chat_id,
            channel_type=channel_type,
            thread_ts=thread_ts,
            raw_thread_ts=raw_thread_ts,
            current_ts=event_ts,
        )
        if file_markers:
            content = "\n".join(part for part in [content, *file_markers] if part)
        if not content and not media_paths:
            return

        try:
            await self._handle_message(
                sender_id=sender_id,
                chat_id=chat_id,
                content=content,
                media=media_paths,
                metadata={
                    "slack": {
                        "event": event,
                        "thread_ts": thread_ts,
                        "channel_type": channel_type,
                    },
                },
                session_key=session_key,
            )
        except Exception:
            logger.exception("Error handling Slack message from {}", sender_id)

    async def _download_slack_file(self, file_info: dict[str, Any]) -> tuple[str | None, str]:
        """Download a Slack private file to the local media directory."""
        file_id = str(file_info.get("id") or "file")
        name = str(
            file_info.get("name")
            or file_info.get("title")
            or file_info.get("id")
            or "slack-file"
        )
        marker_type = "image" if str(file_info.get("mimetype") or "").startswith("image/") else "file"
        marker = f"[{marker_type}: {name}]"
        url = str(file_info.get("url_private_download") or file_info.get("url_private") or "")
        if not url:
            return None, self._download_failure_marker(marker_type, name, "missing download url")
        if not self.config.bot_token:
            return None, self._download_failure_marker(marker_type, name, "missing bot token")

        filename = safe_filename(f"{file_id}_{name}")
        path = Path(get_media_dir("slack")) / filename
        try:
            async with httpx.AsyncClient(timeout=SLACK_DOWNLOAD_TIMEOUT, follow_redirects=True) as client:
                response = await client.get(
                    url,
                    headers={"Authorization": f"Bearer {self.config.bot_token}"},
                )
                response.raise_for_status()
            if self._looks_like_html_download(response):
                raise ValueError("Slack returned HTML instead of file content")
            path.write_bytes(response.content)
            return str(path), marker
        except Exception as e:
            logger.warning("Failed to download Slack file {}: {}", file_id, e)
            return None, self._download_failure_marker(marker_type, name, "download failed")

    @staticmethod
    def _download_failure_marker(marker_type: str, name: str, reason: str) -> str:
        return (
            f"[{marker_type}: {name}: {reason}; not available to nanobot. "
            "Check Slack files:read scope, reinstall the Slack app, and ensure the bot can access the file.]"
        )

    @staticmethod
    def _looks_like_html_download(response: httpx.Response) -> bool:
        content_type = response.headers.get("content-type", "").lower()
        if "text/html" in content_type:
            return True
        preview = response.content[:256].lstrip().lower()
        return preview.startswith(_HTML_DOWNLOAD_PREFIXES)

    async def _on_block_action(self, client: SocketModeClient, req: SocketModeRequest) -> None:
        """Handle button clicks from ask_user blocks."""
        await client.send_socket_mode_response(SocketModeResponse(envelope_id=req.envelope_id))
        payload = req.payload or {}
        actions = payload.get("actions") or []
        if not actions:
            return
        value = str(actions[0].get("value") or "")
        user_info = payload.get("user") or {}
        sender_id = str(user_info.get("id") or "")
        channel_info = payload.get("channel") or {}
        chat_id = str(channel_info.get("id") or "")
        if not sender_id or not chat_id or not value:
            return
        message_info = payload.get("message") or {}
        thread_ts = message_info.get("thread_ts") or message_info.get("ts")
        channel_type = self._infer_channel_type(chat_id)
        if not self._is_allowed(sender_id, chat_id, channel_type):
            return
        session_key = f"slack:{chat_id}:{thread_ts}" if thread_ts else None
        try:
            await self._handle_message(
                sender_id=sender_id,
                chat_id=chat_id,
                content=value,
                metadata={"slack": {"thread_ts": thread_ts, "channel_type": channel_type}},
                session_key=session_key,
            )
        except Exception:
            logger.exception("Error handling Slack button click from {}", sender_id)

    async def _with_thread_context(
        self,
        text: str,
        *,
        chat_id: str,
        channel_type: str,
        thread_ts: str | None,
        raw_thread_ts: str | None,
        current_ts: str | None,
    ) -> str:
        """Include thread history the first time the bot is pulled into a Slack thread."""
        del channel_type  # DM and channel threads are both fetched via conversations.replies
        client = self._get_client(chat_id)
        if (
            not self.config.include_thread_context
            or not client
            or not raw_thread_ts
            or not thread_ts
            or current_ts == thread_ts
        ):
            return text

        key = f"{chat_id}:{thread_ts}"
        if key in self._thread_context_attempted:
            return text
        if len(self._thread_context_attempted) >= self._THREAD_CONTEXT_CACHE_LIMIT:
            self._thread_context_attempted.clear()
        self._thread_context_attempted.add(key)

        try:
            response = await client.conversations_replies(
                channel=chat_id,
                ts=thread_ts,
                limit=max(1, self.config.thread_context_limit),
            )
        except Exception as e:
            logger.warning("Slack thread context unavailable for {}: {}", key, e)
            return text

        lines = self._format_thread_context(
            response.get("messages", []),
            current_ts=current_ts,
        )
        if not lines:
            return text
        return "Slack thread context before this mention:\n" + "\n".join(lines) + f"\n\nCurrent message:\n{text}"

    def _format_thread_context(self, messages: list[dict[str, Any]], *, current_ts: str | None) -> list[str]:
        lines: list[str] = []
        for item in messages:
            if item.get("ts") == current_ts:
                continue
            if item.get("subtype"):
                continue
            sender = str(item.get("user") or item.get("bot_id") or "unknown")
            is_bot = sender in self._all_bot_user_ids
            label = "bot" if is_bot else f"<@{sender}>"
            text = str(item.get("text") or "").strip()
            if not text:
                continue
            text = self._strip_bot_mention(text)
            if len(text) > 500:
                text = text[:500] + "…"
            lines.append(f"- {label}: {text}")
        return lines

    @staticmethod
    def _build_button_blocks(text: str, buttons: list[list[str]]) -> list[dict[str, Any]]:
        """Build Slack Block Kit blocks with action buttons for ask_user choices."""
        blocks: list[dict[str, Any]] = [
            {"type": "section", "text": {"type": "mrkdwn", "text": text[:3000]}},
        ]
        elements = []
        for row in buttons:
            for label in row:
                elements.append({
                    "type": "button",
                    "text": {"type": "plain_text", "text": label[:75]},
                    "value": label[:75],
                    "action_id": f"ask_user_{label[:50]}",
                })
        if elements:
            blocks.append({"type": "actions", "elements": elements[:25]})
        return blocks

    async def _update_react_emoji(self, chat_id: str, ts: str | None) -> None:
        """Clear the processing indicator (typing status or reaction emoji)."""
        client = self._get_client(chat_id)
        if not client or not ts:
            return
        if self.config.typing_status:
            await self._set_typing_status(chat_id, ts, "")
            return
        try:
            await client.reactions_remove(
                channel=chat_id,
                name=self.config.react_emoji,
                timestamp=ts,
            )
        except Exception as e:
            logger.debug("Slack reactions_remove failed: {}", e)
        if self.config.done_emoji:
            try:
                await client.reactions_add(
                    channel=chat_id,
                    name=self.config.done_emoji,
                    timestamp=ts,
                )
            except Exception as e:
                logger.debug("Slack done reaction failed: {}", e)

    async def _set_typing_status(self, channel_id: str, thread_ts: str | None, status: str) -> None:
        """Set or clear the assistant typing status indicator on a thread."""
        client = self._get_client(channel_id)
        if not client or not thread_ts:
            return
        try:
            kwargs: dict = {
                "channel_id": channel_id,
                "thread_ts": thread_ts,
                "status": status,
            }
            if status:
                kwargs["loading_messages"] = [status]
            await client.assistant_threads_setStatus(**kwargs)
        except Exception as e:
            logger.debug("Slack assistant_threads_setStatus failed: {}", e)

    def _is_allowed(self, sender_id: str, chat_id: str, channel_type: str) -> bool:
        if channel_type == "im":
            if not self.config.dm.enabled:
                return False
            if self.config.dm.policy == "allowlist":
                return sender_id in self.config.dm.allow_from
            return True

        # Group / channel messages
        if self.config.group_policy == "allowlist":
            return chat_id in self.config.group_allow_from
        return True

    def _should_respond_in_channel(
        self, event_type: str, text: str, chat_id: str, thread_ts: str | None = None,
    ) -> bool:
        if self.config.group_policy == "open":
            return True
        if self.config.group_policy == "mention":
            if event_type == "app_mention":
                return True
            if thread_ts and thread_ts in self._mention_threads:
                return True
            return any(f"<@{bot_id}>" in text for bot_id in self._all_bot_user_ids)
        if self.config.group_policy == "allowlist":
            return chat_id in self.config.group_allow_from
        return False

    _MAX_MENTION_THREADS = 10_000

    def _track_mention_thread(self, thread_key: str) -> None:
        if thread_key in self._mention_threads:
            self._mention_threads.move_to_end(thread_key)
        else:
            self._mention_threads[thread_key] = None
            if len(self._mention_threads) > self._MAX_MENTION_THREADS:
                self._mention_threads.popitem(last=False)

    def is_allowed(self, sender_id: str) -> bool:
        return True

    @staticmethod
    def _infer_channel_type(chat_id: str) -> str:
        if chat_id.startswith("D"):
            return "im"
        if chat_id.startswith("G"):
            return "group"
        return "channel"

    def _strip_bot_mention(self, text: str) -> str:
        if not text or not self._all_bot_user_ids:
            return text
        for bot_id in sorted(self._all_bot_user_ids, key=len, reverse=True):
            text = re.sub(rf"<@{re.escape(bot_id)}>\s*", "", text)
        return text.strip()

    _TABLE_RE = re.compile(r"(?m)^\|.*\|$(?:\n\|[\s:|-]*\|$)(?:\n\|.*\|$)*")
    _CODE_FENCE_RE = re.compile(r"```[\s\S]*?```")
    _INLINE_CODE_RE = re.compile(r"`[^`]+`")
    _LEFTOVER_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
    _LEFTOVER_HEADER_RE = re.compile(r"^#{1,6}\s+(.+)$", re.MULTILINE)
    _BARE_URL_RE = re.compile(r"(?<![|<])(https?://\S+)")

    @classmethod
    def _to_mrkdwn(cls, text: str) -> str:
        """Convert Markdown to Slack mrkdwn, including tables."""
        if not text:
            return ""
        text = cls._TABLE_RE.sub(cls._convert_table, text)
        return cls._fixup_mrkdwn(slackify_markdown(text)).rstrip("\n")

    @classmethod
    def _fixup_mrkdwn(cls, text: str) -> str:
        """Fix markdown artifacts that slackify_markdown misses."""
        code_blocks: list[str] = []

        def _save_code(m: re.Match) -> str:
            code_blocks.append(m.group(0))
            return f"\x00CB{len(code_blocks) - 1}\x00"

        text = cls._CODE_FENCE_RE.sub(_save_code, text)
        text = cls._INLINE_CODE_RE.sub(_save_code, text)
        text = cls._LEFTOVER_BOLD_RE.sub(r"*\1*", text)
        text = cls._LEFTOVER_HEADER_RE.sub(r"*\1*", text)
        text = cls._BARE_URL_RE.sub(lambda m: m.group(0).replace("&amp;", "&"), text)

        for i, block in enumerate(code_blocks):
            text = text.replace(f"\x00CB{i}\x00", block)
        return text

    @staticmethod
    def _convert_table(match: re.Match) -> str:
        """Convert a Markdown table to a Slack-readable list."""
        lines = [ln.strip() for ln in match.group(0).strip().splitlines() if ln.strip()]
        if len(lines) < 2:
            return match.group(0)
        headers = [h.strip() for h in lines[0].strip("|").split("|")]
        start = 2 if re.fullmatch(r"[|\s:\-]+", lines[1]) else 1
        rows: list[str] = []
        for line in lines[start:]:
            cells = [c.strip() for c in line.strip("|").split("|")]
            cells = (cells + [""] * len(headers))[: len(headers)]
            parts = [f"**{headers[i]}**: {cells[i]}" for i in range(len(headers)) if cells[i]]
            if parts:
                rows.append(" · ".join(parts))
        return "\n".join(rows)
