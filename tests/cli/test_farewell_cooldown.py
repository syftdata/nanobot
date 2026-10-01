"""Unit tests for Slack farewell rate-limiting (SYF2-5804)."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from nanobot.cli.commands import (
    FAREWELL_COOLDOWN,
    _farewell_cooldown_path,
    farewell_on_cooldown,
    record_farewell_sent,
)


def test_farewell_cooldown_path_is_under_workspace_state(tmp_path: Path) -> None:
    assert _farewell_cooldown_path(tmp_path) == tmp_path / "state" / "farewell_sent.json"


def test_no_prior_farewell_is_not_on_cooldown(tmp_path: Path) -> None:
    assert farewell_on_cooldown(tmp_path, "C123") is False


def test_record_then_cooldown_skips_within_window(tmp_path: Path) -> None:
    now = datetime(2026, 9, 30, 21, 0, 0, tzinfo=timezone.utc)
    record_farewell_sent(tmp_path, "C123", now=now)

    assert farewell_on_cooldown(tmp_path, "C123", now=now) is True
    assert farewell_on_cooldown(
        tmp_path, "C123", now=now + FAREWELL_COOLDOWN - timedelta(seconds=1)
    ) is True
    # Different chat is independent
    assert farewell_on_cooldown(tmp_path, "C999", now=now) is False


def test_cooldown_expires_after_15_minutes(tmp_path: Path) -> None:
    now = datetime(2026, 9, 30, 21, 0, 0, tzinfo=timezone.utc)
    record_farewell_sent(tmp_path, "C123", now=now)
    assert (
        farewell_on_cooldown(tmp_path, "C123", now=now + FAREWELL_COOLDOWN + timedelta(seconds=1))
        is False
    )


def test_corrupt_cooldown_file_does_not_skip(tmp_path: Path) -> None:
    path = _farewell_cooldown_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not-json", encoding="utf-8")
    assert farewell_on_cooldown(tmp_path, "C123") is False


def test_record_persists_iso_utc_timestamp(tmp_path: Path) -> None:
    now = datetime(2026, 9, 30, 21, 54, 44, tzinfo=timezone.utc)
    record_farewell_sent(tmp_path, "C071BCXFL12", now=now)
    data = json.loads(_farewell_cooldown_path(tmp_path).read_text(encoding="utf-8"))
    assert data["C071BCXFL12"] == now.isoformat()
