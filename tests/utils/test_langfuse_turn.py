"""Tests for the per-turn Langfuse trace context (utils/langfuse_turn.py)."""

from pathlib import Path

import pytest

from nanobot.utils.langfuse_turn import (
    _channel_of,
    _last_user_text,
    _org_id,
    set_turn_output,
    turn_trace,
)


class TestLastUserText:
    def test_plain_string_content(self):
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "tell me more about ping wu"},
        ]
        assert _last_user_text(msgs) == "tell me more about ping wu"

    def test_takes_most_recent_user_message(self):
        msgs = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "reply"},
            {"role": "user", "content": "second"},
        ]
        assert _last_user_text(msgs) == "second"

    def test_multimodal_list_content(self):
        msgs = [{"role": "user", "content": [{"type": "text", "text": "hi there"}]}]
        assert _last_user_text(msgs) == "hi there"

    def test_none_and_empty(self):
        assert _last_user_text(None) is None
        assert _last_user_text([]) is None
        assert _last_user_text([{"role": "assistant", "content": "x"}]) is None


class TestChannelOf:
    def test_channel_level_key(self):
        assert _channel_of("slack:C076Z3UUFCG", "C0SINK") == "C076Z3UUFCG"

    def test_thread_key_collapses_to_channel(self):
        # Thread replies join the CHANNEL session, so a digest and its threaded
        # replies land in one Langfuse session.
        assert _channel_of("slack:C076Z3UUFCG:1783.5541", "C076Z3UUFCG") == "C076Z3UUFCG"

    def test_cron_sink_chat_id_is_not_used_when_session_key_present(self):
        # Digest cron turns: session_key names the CUSTOMER channel while
        # chat_id is the reply sink — the session key wins.
        assert _channel_of("slack:C0CUSTOMER", "C0B8SINK") == "C0CUSTOMER"

    def test_falls_back_to_chat_id(self):
        assert _channel_of(None, "C0FALLBACK") == "C0FALLBACK"
        assert _channel_of("", "C0FALLBACK") == "C0FALLBACK"


class TestOrgId:
    def test_fleet_layout(self):
        ws = Path("/opt/nanoagent/workspaces/demanddrive.com/workspace")
        assert _org_id(ws) == "demanddrive.com"

    def test_none_workspace(self):
        assert _org_id(None) is None


class TestTurnTrace:
    def test_noop_without_secret_key(self, monkeypatch):
        monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
        # Must yield cleanly with zero langfuse involvement.
        with turn_trace(
            channel="slack",
            chat_id="C1",
            session_key="slack:C1",
            metadata=None,
            workspace=None,
        ):
            ran = True
        assert ran

    def test_noop_when_langfuse_missing(self, monkeypatch):
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
        # Simulate langfuse not installed.
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "langfuse":
                raise ImportError("langfuse not installed")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        with turn_trace(
            channel="slack",
            chat_id="C1",
            session_key="slack:C1",
            metadata=None,
            workspace=None,
        ):
            ran = True
        assert ran

    def test_traced_turn_sets_session_user_and_tags(self, monkeypatch):
        langfuse = pytest.importorskip("langfuse")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")

        captured: dict = {"updates": []}

        class FakeSpan:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def update(self, **kwargs):
                captured["updates"].append(kwargs)

        class FakeClient:
            def start_as_current_observation(self, **kwargs):
                captured["observation"] = kwargs
                return FakeSpan()

        from contextlib import contextmanager

        @contextmanager
        def fake_propagate(**kwargs):
            captured["attrs"] = kwargs
            yield

        monkeypatch.setattr(langfuse, "get_client", lambda: FakeClient())
        monkeypatch.setattr(langfuse, "propagate_attributes", fake_propagate)

        metadata = {
            "slack": {"event": {"user": "U0AB12CD3", "team": "T964JKJ5P"}},
        }
        with turn_trace(
            channel="slack",
            chat_id="C076Z3UUFCG",
            session_key="slack:C076Z3UUFCG:1783.5541",
            metadata=metadata,
            workspace=Path("/opt/nanoagent/workspaces/demanddrive.com/workspace"),
            initial_messages=[{"role": "user", "content": "who is ping wu?"}],
        ) as span:
            # The span is yielded so the caller can attach the final reply.
            set_turn_output(span, "Ping Wu is a Syft cofounder.")

        # Same session key shape the webapp stamps on digest MCP traces.
        assert captured["attrs"]["session_id"] == "slack:demanddrive.com:C076Z3UUFCG"
        assert captured["attrs"]["user_id"] == "U0AB12CD3"
        assert "user_reply" in captured["attrs"]["tags"]
        assert "org:demanddrive.com" in captured["attrs"]["tags"]
        assert captured["observation"]["name"] == "user_reply"
        # Trace-level input/output so Sessions read like a transcript.
        assert captured["observation"]["input"] == "who is ping wu?"
        assert captured["updates"] == [{"output": "Ping Wu is a Syft cofounder."}]

    def test_digest_cron_turn(self, monkeypatch):
        langfuse = pytest.importorskip("langfuse")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")

        captured: dict = {}

        class FakeSpan:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        class FakeClient:
            def start_as_current_observation(self, **kwargs):
                captured["observation"] = kwargs
                return FakeSpan()

        from contextlib import contextmanager

        @contextmanager
        def fake_propagate(**kwargs):
            captured["attrs"] = kwargs
            yield

        monkeypatch.setattr(langfuse, "get_client", lambda: FakeClient())
        monkeypatch.setattr(langfuse, "propagate_attributes", fake_propagate)

        # Shape produced by cron/bound_runner.py for a seeded digest job.
        metadata = {
            "_cron_trigger": {
                "job_id": "syft-daily-summary-C076Z3UUFCG",
                "job_name": "Daily lead-alert summary (C076Z3UUFCG)",
                "run_id": "run-123",
            },
            "slack": {},
        }
        with turn_trace(
            channel="slack",
            chat_id="C0B8SINK",  # reply sink — must NOT become the session
            session_key="slack:C076Z3UUFCG",
            metadata=metadata,
            workspace=Path("/opt/nanoagent/workspaces/demanddrive.com/workspace"),
        ):
            pass

        assert captured["attrs"]["session_id"] == "slack:demanddrive.com:C076Z3UUFCG"
        assert captured["attrs"]["user_id"] == "cron"
        assert "digest" in captured["attrs"]["tags"]
        assert "cron" in captured["attrs"]["tags"]
        assert "user_reply" not in captured["attrs"]["tags"]
        assert captured["observation"]["name"] == "digest_run"
        assert captured["attrs"]["metadata"]["cron_run_id"] == "run-123"

    def test_set_turn_output_is_safe_on_none_and_errors(self):
        # None span (tracing off) → no-op.
        set_turn_output(None, "reply")
        # Span whose update() raises → swallowed.
        class ExplodingSpan:
            def update(self, **kwargs):
                raise RuntimeError("boom")

        set_turn_output(ExplodingSpan(), "reply")
        # Empty output → no update attempted.
        class MustNotUpdate:
            def update(self, **kwargs):
                raise AssertionError("should not be called")

        set_turn_output(MustNotUpdate(), None)
        set_turn_output(MustNotUpdate(), "")

    def test_tracing_failure_never_breaks_the_turn(self, monkeypatch):
        langfuse = pytest.importorskip("langfuse")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")

        def boom():
            raise RuntimeError("langfuse down")

        monkeypatch.setattr(langfuse, "get_client", boom)
        with turn_trace(
            channel="slack",
            chat_id="C1",
            session_key="slack:C1",
            metadata=None,
            workspace=None,
        ):
            ran = True
        assert ran
