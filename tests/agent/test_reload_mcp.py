"""Tests for SIGHUP-triggered MCP reload in AgentLoop."""

from __future__ import annotations

import json
from contextlib import AsyncExitStack
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from nanobot.agent.loop import AgentLoop
from nanobot.agent.tools.base import Tool
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.bus.queue import MessageBus


class _FakeTool(Tool):
    def __init__(self, name: str):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return f"{self._name} tool"

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    async def execute(self, **kwargs: Any) -> Any:
        return kwargs


def _provider(default_model: str = "test-model") -> MagicMock:
    provider = MagicMock()
    provider.get_default_model.return_value = default_model
    provider.generation = SimpleNamespace(max_tokens=4096)
    return provider


def _make_loop(
    tmp_path: Path,
    *,
    mcp_servers: dict | None = None,
    config_path: Path | None = None,
) -> AgentLoop:
    return AgentLoop(
        bus=MessageBus(),
        provider=_provider(),
        workspace=tmp_path,
        model="test-model",
        mcp_servers=mcp_servers,
        config_path=config_path,
    )


# ---------------------------------------------------------------------------
# ToolRegistry.unregister_mcp_tools
# ---------------------------------------------------------------------------


class TestUnregisterMcpTools:
    def test_removes_only_mcp_prefixed_tools(self) -> None:
        registry = ToolRegistry()
        registry.register(_FakeTool("read_file"))
        registry.register(_FakeTool("mcp_syft_query"))
        registry.register(_FakeTool("mcp_hubspot_get"))
        registry.register(_FakeTool("write_file"))

        removed = registry.unregister_mcp_tools()

        assert removed == 2
        assert "read_file" in registry
        assert "write_file" in registry
        assert "mcp_syft_query" not in registry
        assert "mcp_hubspot_get" not in registry

    def test_returns_zero_when_no_mcp_tools(self) -> None:
        registry = ToolRegistry()
        registry.register(_FakeTool("read_file"))

        removed = registry.unregister_mcp_tools()

        assert removed == 0
        assert "read_file" in registry

    def test_invalidates_definition_cache(self) -> None:
        registry = ToolRegistry()
        registry.register(_FakeTool("mcp_syft_query"))
        registry.register(_FakeTool("read_file"))
        _ = registry.get_definitions()
        assert registry._cached_definitions is not None

        registry.unregister_mcp_tools()

        assert registry._cached_definitions is None

    def test_no_cache_invalidation_when_nothing_removed(self) -> None:
        registry = ToolRegistry()
        registry.register(_FakeTool("read_file"))
        _ = registry.get_definitions()
        cached_ref = registry._cached_definitions

        registry.unregister_mcp_tools()

        assert registry._cached_definitions is cached_ref


# ---------------------------------------------------------------------------
# AgentLoop.reload_mcp
# ---------------------------------------------------------------------------


class TestReloadMcp:
    async def test_reload_reconnects_with_fresh_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """reload_mcp should close existing MCP, re-read config, and reconnect."""
        config_path = tmp_path / "config.json"
        config_data = {
            "tools": {
                "mcpServers": {
                    "hubspot": {
                        "command": "npx",
                        "args": ["-y", "@hubspot/mcp-server"],
                        "env": {"PRIVATE_APP_ACCESS_TOKEN": "new-token"},
                    }
                }
            }
        }
        config_path.write_text(json.dumps(config_data))

        loop = _make_loop(
            tmp_path,
            mcp_servers={"old": object()},
            config_path=config_path,
        )
        loop._mcp_connected = True
        loop._mcp_stacks = {"old": AsyncExitStack()}

        connect_calls: list[dict] = []

        async def _fake_connect(servers, _registry):
            connect_calls.append(dict(servers))
            return {name: AsyncExitStack() for name in servers}

        monkeypatch.setattr("nanobot.agent.tools.mcp.connect_mcp_servers", _fake_connect)

        await loop.reload_mcp()

        assert len(connect_calls) == 1
        assert "hubspot" in connect_calls[0]
        assert loop._mcp_connected is True

    async def test_reload_preserves_state_on_config_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If config loading raises, reload_mcp should log and keep existing state."""
        config_path = tmp_path / "config.json"
        config_path.write_text("{}")

        original_server = object()
        loop = _make_loop(
            tmp_path,
            mcp_servers={"syft": original_server},
            config_path=config_path,
        )
        loop._mcp_connected = True

        def _exploding_load(*_args, **_kwargs):
            raise RuntimeError("disk on fire")

        monkeypatch.setattr("nanobot.config.loader.load_config", _exploding_load)

        await loop.reload_mcp()

        assert loop._mcp_connected is True
        assert loop._mcp_servers["syft"] is original_server

    async def test_reload_clears_mcp_tools_before_reconnect(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Old MCP tool wrappers should be removed before reconnecting."""
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({"tools": {"mcpServers": {}}}))

        loop = _make_loop(tmp_path, config_path=config_path)
        loop.tools.register(_FakeTool("mcp_syft_query"))
        loop.tools.register(_FakeTool("read_file"))

        async def _fake_connect(servers, _registry):
            return {}

        monkeypatch.setattr("nanobot.agent.tools.mcp.connect_mcp_servers", _fake_connect)

        await loop.reload_mcp()

        assert "mcp_syft_query" not in loop.tools
        assert "read_file" in loop.tools
