"""Slack channel package."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = [
    "SLACK_MARKDOWN_BLOCK_LEN",
    "SLACK_MAX_MESSAGE_LEN",
    "SlackChannel",
    "SlackConfig",
    "SlackDMConfig",
]

if TYPE_CHECKING:
    from .runtime import SlackChannel, SlackConfig, SlackDMConfig


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(name)
    from . import runtime

    return getattr(runtime, name)
