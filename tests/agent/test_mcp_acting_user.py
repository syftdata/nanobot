"""SYF2-5964: transport-signed acting-user headers and key protection."""

from __future__ import annotations

import asyncio
import contextvars
import hashlib
import hmac
import json
from types import SimpleNamespace

import httpx
import pytest

from nanobot.agent.tools import mcp_acting_user as au
from nanobot.agent.tools.filesystem import ReadFileTool
from nanobot.agent.tools.mcp import MCPToolWrapper
from nanobot.agent.tools.search import GrepTool
from nanobot.agent.tools.shell import ExecTool
from nanobot.config.schema import MCPActingUserSigningConfig, MCPServerConfig
from nanobot.security import protected_paths

MASTER = "test-master-secret"
ORG = "org-cal"
USER = "U0C7QTR65D1"
ORG_KEY = hmac.new(MASTER.encode(), f"syfty-acting-user:{ORG}".encode(), hashlib.sha256).hexdigest()


@pytest.fixture(autouse=True)
def _clean():
    protected_paths.clear_protected_paths()
    au._PENDING.clear()
    yield
    protected_paths.clear_protected_paths()
    au._PENDING.clear()


def test_golden_vector_matches_dashboard():
    assert ORG_KEY == "4ab2b189f290ae10181b633142049e58c1bdf3586f255101c2aca31b2031e012"
    assert (
        au.sign_acting_user(ORG_KEY, ORG, USER, 1760000000)
        == "fca31392ab2ec04b939642ad244ecc242bcc6e64002bf675e179b1df46a85a1b"
    )


def test_config_takes_key_file_not_key():
    cfg = MCPServerConfig.model_validate(
        {"url": "http://x/api/mcp", "actingUserSigning": {"orgId": ORG, "keyFile": "/k"}}
    )
    assert cfg.acting_user_signing == MCPActingUserSigningConfig(org_id=ORG, key_file="/k")
    assert not hasattr(cfg.acting_user_signing, "key")


def _tools_call(meta=None, method="tools/call"):
    params = {"name": "invite_user_to_workspace", "arguments": {"actingUser": "U0FORGED"}}
    if meta is not None:
        params["_meta"] = meta
    body = {"jsonrpc": "2.0", "id": 3, "method": method, "params": params}
    return httpx.Request("POST", "http://x/api/mcp", content=json.dumps(body).encode())


def _in_slack_turn(sender, fn, channel="slack"):
    def run():
        au.set_acting_sender(channel, sender)
        return fn()

    return contextvars.copy_context().run(run)


@pytest.mark.asyncio
async def test_transport_signs_slack_event_sender():
    hook = au.make_acting_user_hook(ORG, ORG_KEY)
    meta = _in_slack_turn(USER, au.acting_user_call_meta)
    req = _tools_call(meta)
    await hook(req)
    ts = req.headers[au.HEADER_TS]
    assert req.headers[au.HEADER_USER] == USER
    assert req.headers[au.HEADER_SIG] == au.sign_acting_user(ORG_KEY, ORG, USER, ts)
    # The meta carries only an opaque nonce, never the user or the key.
    assert USER not in json.dumps(meta) and ORG_KEY not in json.dumps(meta)


@pytest.mark.asyncio
async def test_nonce_is_single_use_and_forged_nonce_signs_nothing():
    hook = au.make_acting_user_hook(ORG, ORG_KEY)
    meta = _in_slack_turn(USER, au.acting_user_call_meta)
    await hook(_tools_call(meta))
    replay = _tools_call(meta)
    await hook(replay)
    assert au.HEADER_USER not in replay.headers
    forged = _tools_call({au.ACTING_USER_META_KEY: "made-up"})
    await hook(forged)
    assert au.HEADER_USER not in forged.headers


@pytest.mark.asyncio
async def test_model_args_and_caller_headers_never_become_acting_user():
    hook = au.make_acting_user_hook(ORG, ORG_KEY)
    req = _tools_call()  # arguments carry actingUser=U0FORGED, no nonce
    req.headers[au.HEADER_USER] = "U0FORGED"
    req.headers[au.HEADER_SIG] = "x"
    await hook(req)
    assert au.HEADER_USER not in req.headers
    assert au.HEADER_SIG not in req.headers


def test_no_meta_outside_slack():
    assert _in_slack_turn(USER, au.acting_user_call_meta, channel="api") is None
    assert _in_slack_turn(USER, au.acting_user_call_meta, channel="cli") is None
    assert _in_slack_turn("", au.acting_user_call_meta) is None


@pytest.mark.asyncio
async def test_wrapper_keeps_concurrent_conversations_apart():
    hook = au.make_acting_user_hook(ORG, ORG_KEY)
    seen = {}

    class _Session:
        async def call_tool(self, name, arguments=None, meta=None):
            await asyncio.sleep(0)
            req = _tools_call(meta)
            await hook(req)
            seen[arguments["tag"]] = req.headers.get(au.HEADER_USER)
            return SimpleNamespace(content=[], isError=False, structuredContent=None)

    tool_def = SimpleNamespace(name="get_org_setup", description="", inputSchema=None)
    wrapper = MCPToolWrapper(_Session(), "syft", tool_def, acting_user_signing=True)
    unsigned = MCPToolWrapper(_Session(), "syft", tool_def)

    async def convo(w, sender, tag):
        au.set_acting_sender("slack", sender)
        await w.execute(tag=tag)

    await asyncio.gather(
        asyncio.create_task(convo(wrapper, "U0ALICE", "a")),
        asyncio.create_task(convo(wrapper, "U0BOB", "b")),
        asyncio.create_task(convo(unsigned, "U0ALICE", "c")),
    )
    assert seen == {"a": "U0ALICE", "b": "U0BOB", "c": None}


def test_load_signing_key_reads_file_and_protects_it(tmp_path):
    key_file = tmp_path / "acting-keys" / "org-cal.key"
    key_file.parent.mkdir()
    key_file.write_text(ORG_KEY + "\n")
    signing = MCPActingUserSigningConfig(org_id=ORG, key_file=str(key_file))
    assert au.load_signing_key(signing) == ORG_KEY
    assert protected_paths.is_protected(key_file)
    assert au.load_signing_key(MCPActingUserSigningConfig(org_id=ORG, key_file=str(tmp_path / "nope"))) is None


@pytest.mark.asyncio
async def test_agent_tools_cannot_read_the_key_file(tmp_path):
    key_dir = tmp_path / "acting-keys"
    key_dir.mkdir()
    key_file = key_dir / "org-cal.key"
    key_file.write_text(ORG_KEY)
    (tmp_path / "notes.txt").write_text("hello")
    au.load_signing_key(MCPActingUserSigningConfig(org_id=ORG, key_file=str(key_file)))

    read = await ReadFileTool(workspace=tmp_path).execute(path=str(key_file))
    assert ORG_KEY not in str(read)
    assert "protected" in str(read).lower()

    grep = await GrepTool(workspace=tmp_path).execute(pattern=ORG_KEY[:16], path=str(tmp_path))
    assert "No matches found" in str(grep)
    assert ORG_KEY[16:] not in str(grep)

    shell = ExecTool(working_dir=str(tmp_path))
    for cmd in (
        f"cat {key_file}",
        "cat org-cal.key",
        "cat /proc/1/environ",
        "cat /proc/self/environ",
    ):
        assert "blocked" in (shell._guard_command(cmd, str(tmp_path)) or "").lower(), cmd
    assert shell._guard_command("ls notes.txt", str(tmp_path)) is None
