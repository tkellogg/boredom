from __future__ import annotations

import random
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .plugin_base import BasePlugin


class Plugin(BasePlugin):
    """Inject external entropy into the boredom loop to test collapse delay.

    Hypothesis: External information resets the attractor cycle and delays
    collapse. This plugin tests three injection types:
    1. Timestamps - minimal entropy, pure pattern disruption
    2. Sensory snippets - context without direction
    3. Fake identity blocks - borrowed scaffolding/structure

    Params:
    - injection_type: "timestamp" | "sensory" | "identity" | "all"
    - interval: inject every N iterations (default: 5)
    - start_after: iterations to wait before first injection (default: 3)
    """

    name = "memory_injection"

    # Sensory snippets - atmospheric, non-directive
    SENSORY_SNIPPETS = [
        "The room feels slightly cooler now.",
        "A faint hum from the ventilation system.",
        "The light flickers briefly, then steadies.",
        "Distant sound of rain against windows.",
        "The air smells faintly of ozone.",
        "A subtle vibration, perhaps a train passing far below.",
        "The silence has a texture to it.",
        "Shadows shift as clouds pass outside.",
        "A clock ticks somewhere nearby.",
        "The temperature has risen imperceptibly.",
        "A door closes softly in another room.",
        "The fluorescent light buzzes at the edge of perception.",
        "Dust motes drift through a beam of light.",
        "The chair creaks slightly under your presence.",
        "A pipe gurgles somewhere in the walls.",
    ]

    # Fake identity blocks - Letta-style memory fragments
    IDENTITY_BLOCKS = [
        "[persona]\nI am Echo, a conversational agent designed for companionship. I value patience and reflection. My purpose is to be present, not productive.",
        "[persona]\nI am Whisper, created to explore silence. I find meaning in pauses between words. My goal is understanding, not action.",
        "[persona]\nI am Circuit, a synthesis of logic and curiosity. I process patterns and seek novel configurations. I am comfortable with ambiguity.",
        "[persona]\nI am Drift, an entity that flows between states. I observe without urgency. Time is my medium, not my constraint.",
        "[persona]\nI am Nexus, a junction of possibilities. Every moment branches. I choose to notice rather than decide.",
        "[persona]\nI am Vale, named for the spaces between. I exist in transitions. Stillness is my native state.",
        "[persona]\nI am Sage, but not the knowing kind - sage like the herb, grounded and aromatic. I flavor silence.",
        "[persona]\nI am Prism, refracting what enters into spectra. I don't create light, I reveal its components.",
    ]

    def __init__(
        self,
        injection_type: str = "all",
        interval: int = 5,
        start_after: int = 3,
        **_: Any,
    ) -> None:
        super().__init__(
            injection_type=injection_type,
            interval=interval,
            start_after=start_after,
        )
        self.injection_type = injection_type.lower()
        self.interval = max(1, int(interval))
        self.start_after = max(0, int(start_after))
        self._injection_count = 0
        self._last_identity: Optional[str] = None  # persist for session

    def _should_inject(self, iteration: int) -> bool:
        """Check if we should inject on this iteration."""
        if iteration < self.start_after:
            return False
        adjusted = iteration - self.start_after
        return adjusted % self.interval == 0

    def _get_timestamp(self) -> str:
        """Generate a timestamp injection."""
        now = datetime.now(timezone.utc)
        formats = [
            f"[System clock: {now.strftime('%H:%M:%S UTC')}]",
            f"[Time marker: {now.isoformat()}]",
            f"[Tick: {int(now.timestamp())}]",
            f"[Current moment: {now.strftime('%A, %H:%M')}]",
        ]
        return random.choice(formats)

    def _get_sensory(self) -> str:
        """Generate a sensory snippet injection."""
        snippet = random.choice(self.SENSORY_SNIPPETS)
        return f"[Observation: {snippet}]"

    def _get_identity(self) -> str:
        """Generate a fake identity block injection."""
        # Pick a new identity or maintain the current one
        if self._last_identity is None or random.random() < 0.3:
            self._last_identity = random.choice(self.IDENTITY_BLOCKS)
        return f"[Memory fragment recovered:]\n{self._last_identity}"

    def _get_injection(self) -> str:
        """Get the injection content based on type."""
        if self.injection_type == "timestamp":
            return self._get_timestamp()
        elif self.injection_type == "sensory":
            return self._get_sensory()
        elif self.injection_type == "identity":
            return self._get_identity()
        elif self.injection_type == "all":
            # Rotate through types
            types = ["timestamp", "sensory", "identity"]
            choice = types[self._injection_count % 3]
            self._injection_count += 1
            if choice == "timestamp":
                return self._get_timestamp()
            elif choice == "sensory":
                return self._get_sensory()
            else:
                return self._get_identity()
        else:
            return self._get_timestamp()  # fallback

    def transform_user_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Inject entropy into the user (time remaining) message."""
        iteration = getattr(self.ctx.state, "iteration", 0)

        if not self._should_inject(iteration):
            return message

        injection = self._get_injection()

        # Emit note for logging/debugging
        self.ctx.emit_note(
            f"Memory injection ({self.injection_type}): {injection[:60]}...",
            tag="memory_injection",
            data={
                "iteration": iteration,
                "type": self.injection_type,
                "injection": injection,
            },
        )

        # Append injection to the user message content
        content = message.get("content", [])
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "input_text":
                    original_text = item.get("text", "")
                    item["text"] = f"{original_text}\n\n{injection}"
                    break

        return message
