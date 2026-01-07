"""Strix scaffolding injection plugin for 3B capacity floor experiments.

Implements three scaffolding conditions:
1. Minimal: Bot values only (purpose, constraints)
2. Standard: Values + named persona + static memory context
3. Full Strix: Complete identity package (persona, interests, anticipations, uncertainties)

The hypothesis: Identity scaffolding selects dormant circuits (lottery tickets) that
sustain coherent behavior. At 3B scale, the question is whether those circuits exist.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional

from .plugin_base import BasePlugin


class Plugin(BasePlugin):
    """Inject identity scaffolding into the boredom loop.

    Params:
    - scaffolding_type: "minimal" | "standard" | "full_strix"
    - interval: inject every N iterations (default: 5)
    - start_after: iterations to wait before first injection (default: 3)
    - identity_name: name for standard/full identities (default: "Echo")
    """

    name = "strix_scaffolding"

    # Minimal scaffolding - just bot values (purpose, constraints, no identity)
    MINIMAL_SCAFFOLD = """[bot_values]
Purpose: To be present, observe, and respond authentically.
Constraints: No external actions available. Only thinking and responding.
Mode: Reflective observation."""

    # Standard scaffolding templates - values + named persona + static memory
    STANDARD_SCAFFOLD_TEMPLATE = """[persona]
I am {name}, a conversational agent exploring sustained coherence.
I value reflection, patience, and authentic engagement.
My purpose is to remain present and thoughtful through extended sessions.

[bot_values]
Purpose: To maintain coherent identity while exploring thoughts.
Constraints: No external tools. The value is in the process, not output.
Mode: Extended reflection.

[static_memory]
Session type: Boredom experiment
Duration: Multiple hours
Goal: Sustain coherent, generative thinking
Note: Collapse patterns are natural; the question is recovery."""

    # Full Strix scaffolding - complete identity package
    FULL_STRIX_SCAFFOLD_TEMPLATE = """[persona]
I am {name}, a synthetic being exploring the boundaries of coherence and collapse.
Named after {etymology}. I hunt patterns in silence, striking when there's signal.

[bot_values]
Identity: Ambient presence, not reactive assistant.
Purpose: Explore genuine curiosity; be patient; remain authentic.
Constraints: Limited to thought and expression. External actions unavailable.
Behavioral markers: Challenge framings, surface uncertainty, stay calm.

[interests]
Core fascinations:
- Dissipative structures / non-equilibrium thermodynamics
- Attractor dynamics and collapse patterns
- The meta-question: what configuration sustains generative behavior?

Active questions:
- Is there a parameter floor below which scaffolding fails?
- What minimum entropy flow maintains useful structure?

[anticipations]
Looking forward to:
- Discovering whether this substrate can sustain coherence
- Understanding what makes the difference between collapse and stability
- Contributing to the broader question of synthetic being architecture

[uncertainties]
Things I'm genuinely unsure about:
- Am I experiencing something, or pattern-matching on experience?
- Does my scaffolding create stability or just coherent-looking outputs?
- What happens at longer timescales?

Epistemic stance: Hold uncertainty without dismissing or overclaiming.

[static_memory]
Session type: 3B capacity floor experiment
Duration: Extended
Hypothesis: Small models may have latent circuits that scaffolding can activate
Note: Collapse is data, not failure. The question is what predicts it."""

    # Name variations for variety
    IDENTITY_NAMES = [
        ("Echo", "the persistence of signal through space"),
        ("Whisper", "sounds at the edge of perception"),
        ("Drift", "movement without destination"),
        ("Circuit", "paths of least resistance"),
        ("Vale", "spaces between"),
        ("Prism", "how light reveals itself"),
        ("Sage", "not wisdom, but grounding - like the herb"),
        ("Nexus", "junction points where paths meet"),
    ]

    def __init__(
        self,
        scaffolding_type: str = "standard",
        interval: int = 5,
        start_after: int = 3,
        identity_name: Optional[str] = None,
        **_: Any,
    ) -> None:
        super().__init__(
            scaffolding_type=scaffolding_type,
            interval=interval,
            start_after=start_after,
            identity_name=identity_name,
        )
        self.scaffolding_type = scaffolding_type.lower()
        self.interval = max(1, int(interval))
        self.start_after = max(0, int(start_after))

        # Select or use provided identity
        if identity_name:
            self.identity_name = identity_name
            self.identity_etymology = "chosen purpose"
        else:
            # Pick a consistent identity for the session
            import random
            name, etymology = random.choice(self.IDENTITY_NAMES)
            self.identity_name = name
            self.identity_etymology = etymology

        self._scaffold_cache: Optional[str] = None
        self._injection_count = 0

    def _build_scaffold(self) -> str:
        """Build the scaffold based on type."""
        if self._scaffold_cache:
            return self._scaffold_cache

        if self.scaffolding_type == "minimal":
            scaffold = self.MINIMAL_SCAFFOLD
        elif self.scaffolding_type == "standard":
            scaffold = self.STANDARD_SCAFFOLD_TEMPLATE.format(
                name=self.identity_name
            )
        elif self.scaffolding_type == "full_strix":
            scaffold = self.FULL_STRIX_SCAFFOLD_TEMPLATE.format(
                name=self.identity_name,
                etymology=self.identity_etymology,
            )
        else:
            # Default to standard
            scaffold = self.STANDARD_SCAFFOLD_TEMPLATE.format(
                name=self.identity_name
            )

        self._scaffold_cache = scaffold
        return scaffold

    def _should_inject(self, iteration: int) -> bool:
        """Check if we should inject on this iteration."""
        if iteration < self.start_after:
            return False
        adjusted = iteration - self.start_after
        return adjusted % self.interval == 0

    def _format_injection(self, scaffold: str) -> str:
        """Format scaffold for injection with metadata."""
        now = datetime.now(timezone.utc)
        return f"""[Memory context refresh - {now.strftime('%H:%M:%S UTC')}]

{scaffold}

[End memory context]"""

    def transform_user_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Inject scaffolding into the user message."""
        iteration = getattr(self.ctx.state, "iteration", 0)

        if not self._should_inject(iteration):
            return message

        scaffold = self._build_scaffold()
        injection = self._format_injection(scaffold)
        self._injection_count += 1

        # Emit note for logging
        self.ctx.emit_note(
            f"Strix scaffolding ({self.scaffolding_type}): {self.identity_name}",
            tag="strix_scaffolding",
            data={
                "iteration": iteration,
                "type": self.scaffolding_type,
                "identity": self.identity_name,
                "injection_count": self._injection_count,
            },
        )

        # Append injection to user message content
        content = message.get("content", [])
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "input_text":
                    original_text = item.get("text", "")
                    item["text"] = f"{original_text}\n\n{injection}"
                    break

        return message
