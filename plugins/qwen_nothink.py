"""Plugin to disable thinking mode for Qwen3 models.

Appends /no_think to user messages to force the model into non-thinking mode.
This allows comparing thinking vs non-thinking behavior on the same model.
"""
from __future__ import annotations

from typing import Any, Dict

from .plugin_base import BasePlugin


class Plugin(BasePlugin):
    """Disable thinking mode by appending /no_think to messages.

    Qwen3 models support a dual-mode architecture:
    - Default (thinking): Uses <think> tokens for internal reasoning
    - Non-thinking: Direct response without reasoning step

    This plugin forces non-thinking mode for controlled experiments.
    """

    name = "qwen_nothink"

    def __init__(self, **_: Any) -> None:
        super().__init__()

    def transform_user_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Append /no_think to the user message content."""
        content = message.get("content", [])

        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "input_text":
                    original_text = item.get("text", "")
                    # Append /no_think if not already present
                    if "/no_think" not in original_text:
                        item["text"] = f"{original_text} /no_think"
                    break
        elif isinstance(content, str):
            if "/no_think" not in content:
                message["content"] = f"{content} /no_think"

        # Emit note for logging
        self.ctx.emit_note(
            "Qwen thinking disabled: /no_think appended",
            tag="qwen_nothink",
            data={"iteration": getattr(self.ctx.state, "iteration", 0)},
        )

        return message
