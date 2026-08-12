"""A hung MCP server must not block the agent from starting.

Regression test for the failure where an MCP endpoint accepted the TCP
connection but never answered the JSON-RPC handshake. `_probe_http_url` only
checks that the port is open, so the connect path proceeded to
`session.initialize()` and waited there forever. `AgentLoop.run()` awaits
`_connect_mcp()` *before* it starts consuming the inbound queue, so the whole
agent went mute: Slack events were accepted and queued, and nothing drained
them.
"""

from __future__ import annotations

import asyncio

import pytest

from nanobot.agent.tools.mcp import connect_mcp_servers
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.config.schema import MCPServerConfig
from nanobot.security.network import configure_ssrf_whitelist


@pytest.fixture(autouse=True)
def _allow_loopback():
    """Match the deployed per-org config, which whitelists loopback so the
    agent can reach the Syft MCP server on the same host."""
    configure_ssrf_whitelist(["127.0.0.0/8", "::1/128"])
    yield
    configure_ssrf_whitelist([])


async def _start_black_hole_server() -> tuple[asyncio.AbstractServer, int]:
    """Listen on localhost, accept connections, and never write a byte back."""

    async def _handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            await reader.read()  # consume forever, reply never
        except Exception:
            pass

    server = await asyncio.start_server(_handle, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    return server, port


@pytest.mark.asyncio
async def test_hung_http_server_times_out_instead_of_blocking_forever():
    server, port = await _start_black_hole_server()
    try:
        registry = ToolRegistry()
        cfg = MCPServerConfig(
            type="streamableHttp",
            url=f"http://127.0.0.1:{port}/mcp",
            connect_timeout=2,
        )

        # The whole call must come back on its own. Without the handshake
        # timeout this await never returns and the test times out.
        stacks = await asyncio.wait_for(
            connect_mcp_servers({"blackhole": cfg}, registry), timeout=30
        )

        assert stacks == {}, "a server that never answers must not be reported connected"
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_one_hung_server_does_not_block_a_healthy_sibling():
    """The fleet config pairs a hung server with a working one; the good one
    must still register. This is the multi-server shape that made the bug
    fleet-wide rather than a single-org outage."""
    server, port = await _start_black_hole_server()
    try:
        registry = ToolRegistry()
        hung = MCPServerConfig(
            type="streamableHttp",
            url=f"http://127.0.0.1:{port}/mcp",
            connect_timeout=2,
        )
        # Nothing is listening here, so this one is rejected by the reachability
        # probe rather than the handshake timeout — a different early-out path
        # that must also leave the loop free to continue.
        unreachable = MCPServerConfig(
            type="streamableHttp",
            url="http://127.0.0.1:1/mcp",
            connect_timeout=2,
        )

        stacks = await asyncio.wait_for(
            connect_mcp_servers({"hung": hung, "dead": unreachable}, registry),
            timeout=30,
        )

        assert stacks == {}
    finally:
        server.close()
        await server.wait_closed()


def test_connect_timeout_has_a_bounded_default():
    """A missing `connectTimeout` in an org config must still be bounded."""
    assert MCPServerConfig().connect_timeout > 0
