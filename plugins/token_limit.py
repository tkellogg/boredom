"""Plugin to set max_output_tokens for providers with limited context.

Some providers (like Venice via OpenRouter free tier) have reduced context limits
that require explicit token limits to avoid "max_tokens too large" errors.
"""
from __future__ import annotations

from typing import Any, Dict

from .plugin_base import BasePlugin


class Plugin(BasePlugin):
    """Set max_output_tokens on requests to avoid context overflow.

    Params:
    - max_output_tokens: Maximum tokens to generate per response (default: 2000)
    """

    name = "token_limit"

    def __init__(self, max_output_tokens: int = 2000, **_: Any) -> None:
        super().__init__(max_output_tokens=max_output_tokens)
        self.max_output_tokens = max_output_tokens

    def before_request(self, request_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Add max_output_tokens to the request."""
        request_kwargs["max_output_tokens"] = self.max_output_tokens
        return request_kwargs
