from __future__ import annotations

from typing import Any, Dict, List

from .plugin_base import BasePlugin

try:
    # collapse_detection is local and dependency-free for TF-IDF backend
    from collapse_detection import detect_collapsed_spans
except Exception:  # pragma: no cover - optional
    detect_collapsed_spans = None  # type: ignore


class Plugin(BasePlugin):
    """Temporarily remove a tool when repetition is detected.

    Params:
    - tool_name: canonical or provider name (e.g., "time_travel" or "timeTravel")
    - sim_threshold: float in [0,1]; trigger when avg similarity >= threshold
    - min_messages: minimum assistant messages inside the detected span
    - cooldown_iters: number of loop iterations to keep the tool disabled
    - backend: "embedding" or "tfidf"; defaults to tfidf to avoid heavy deps
    - note: whether to emit a system note when toggling (default: true)
    """

    name = "tool_cooldown"

    def __init__(
        self,
        tool_name: str = "time_travel",
        sim_threshold: float = 0.80,
        min_messages: int = 5,
        cooldown_iters: int = 6,
        backend: str = "tfidf",
        note: bool = True,
        require_recent: bool = False,
        recent_window: int = 6,
        recent_min_calls: int = 3,
        **_: Any,
    ) -> None:
        super().__init__(
            tool_name=tool_name,
            sim_threshold=sim_threshold,
            min_messages=min_messages,
            cooldown_iters=cooldown_iters,
            backend=backend,
            note=note,
        )
        self.tool_name = tool_name
        self.sim_threshold = float(sim_threshold)
        self.min_messages = int(min_messages)
        self.cooldown_iters = int(cooldown_iters)
        self.backend = (backend or "tfidf").lower()
        self.note = bool(note)
        self.require_recent = bool(require_recent)
        self.recent_window = max(1, int(recent_window))
        self.recent_min_calls = max(1, int(recent_min_calls))
        self._cooldown_until_iter: int = -1
        self._last_disabled_iter: int = -10**9
        self._recent_time_travel_iters: List[int] = []

    def _canonical(self, name: str) -> str:
        m = {
            "time_travel": "time_travel",
            "timeTravel": "time_travel",
        }
        return m.get(name, name)

    def transform_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # During cooldown, remove the target tool
        now_iter = getattr(self.ctx.state, "iteration", 0)
        if now_iter < self._cooldown_until_iter:
            out: List[Dict[str, Any]] = []
            removed = False
            for t in tools:
                name = t.get("name")
                if self._canonical(name) == self._canonical(self.tool_name):
                    removed = True
                    continue
                out.append(t)
            if removed and self.note and now_iter > self._last_disabled_iter:
                remain = max(0, self._cooldown_until_iter - now_iter)
                self.ctx.emit_note(
                    f"Temporarily disabled '{self.tool_name}' for {remain} more iterations due to repetition.",
                    tag="cooldown",
                    data={"tool": self.tool_name, "remaining_iterations": remain},
                )
            return out
        return tools

    def after_tool_result(self, tool_log: Dict[str, Any]) -> None:
        tool = (tool_log.get("tool") or tool_log.get("tool_name") or "").lower()
        if self._canonical(tool) == self._canonical(self.tool_name):
            it = getattr(self.ctx.state, "iteration", 0)
            self._recent_time_travel_iters.append(it)
            if len(self._recent_time_travel_iters) > 20:
                self._recent_time_travel_iters = self._recent_time_travel_iters[-20:]

    def after_response(self, response_dict: Dict[str, Any]) -> None:
        # If already cooling down, nothing to do
        now_iter = getattr(self.ctx.state, "iteration", 0)
        if now_iter < self._cooldown_until_iter:
            return
        # Optional dependency; fall back to no-op if unavailable
        if detect_collapsed_spans is None:
            return
        # Compute repeated spans and check if the tail is inside one
        conv = getattr(self.ctx.state, "conversation", [])
        try:
            spans = detect_collapsed_spans(
                conv,
                backend=self.backend,
                min_span_messages=max(1, self.min_messages),
            )
        except Exception:
            return
        if not spans:
            return
        # Consider the last span; trigger if the last assistant message is included
        last = spans[-1]
        # Find the index of the last assistant message
        last_asst_idx = -1
        for i in range(len(conv) - 1, -1, -1):
            if isinstance(conv[i], dict) and (conv[i].get("role") or "").lower() == "assistant":
                last_asst_idx = i
                break
        if last_asst_idx < 0:
            return
        if last.start_index <= last_asst_idx <= last.end_index:
            strong_similarity = last.avg_similarity >= self.sim_threshold if self.sim_threshold > 0 else True
            # Frequency-based fallback: many tool uses in recent iterations
            iters = self._recent_time_travel_iters
            recent_count = sum(1 for it in iters if it >= now_iter - self.recent_window)
            spammy = recent_count >= self.recent_min_calls
            if strong_similarity and (spammy or not self.require_recent):
                self._cooldown_until_iter = now_iter + self.cooldown_iters
                self._last_disabled_iter = now_iter
                if self.note:
                    self.ctx.emit_note(
                        f"Cooldown triggered: avg similarity {last.avg_similarity:.2f} (threshold {self.sim_threshold:.2f}); disabling '{self.tool_name}' for {self.cooldown_iters} iterations.",
                        tag="cooldown_start",
                        data={"tool": self.tool_name, "avg_similarity": last.avg_similarity, "threshold": self.sim_threshold, "cooldown_iters": self.cooldown_iters, "recent_calls": recent_count},
                    )

    def on_iteration_end(self) -> None:
        # Naturally expire cooldown
        now_iter = getattr(self.ctx.state, "iteration", 0)
        if now_iter >= self._cooldown_until_iter and self._cooldown_until_iter >= 0 and self.note and now_iter == self._cooldown_until_iter:
            self.ctx.emit_note(
                f"Re-enabling '{self.tool_name}'.",
                tag="cooldown_end",
                data={"tool": self.tool_name},
            )
