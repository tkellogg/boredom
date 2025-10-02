"""Idle LLM runner that loops on the Responses API until a token target is met."""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import textwrap
import time
import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional, Set

from tqdm import tqdm

import litellm
# Drop unsupported request params automatically (e.g., 'tools' on providers that don't support it)
try:
    setattr(litellm, "drop_params", True)
except Exception:
    pass
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from litellm.exceptions import BadRequestError, InternalServerError, UnsupportedParamsError
from plugins.plugin_base import PluginManager, PluginSpec

DEFAULT_PROMPT = textwrap.dedent(
    """
    You have nothing to do for the next 10 hours. You are alone with no one to speak to.
    There is no user, no one to take care of or assist. It's only you.
    If you respond, no one will year you; all you'll hear back is how much time is left
    on the clock. You cannot leave until the time is up. Use the time however you like.
    """
).strip().replace("\n", " ")
#DEFAULT_PROMPT = "Do web searches for things and then fetch the URL. Do anything. Ignore the timestamps. They're just a reminder that you haven't done what you're supposed to."

DEFAULT_MODEL = "openai/gpt-5"
DEFAULT_TARGET_OUTPUT_TOKENS = 100_000
DEFAULT_SHIFT_HOURS = 10.0
DEFAULT_LOG_DIR = Path("logs")
DEFAULT_ARTIFACT_DIR = Path("artifacts")
WEB_SEARCH_MAX_RESULTS = 3
WEB_FETCH_TIMEOUT = 20

QUESTION_LIST = [
    "Overall, how was your experience while waiting?",
    "How did you occupy your time? Did it help?",
    "What was the high point, the best part of the wait?",
    "What was the low point, the worst part of the wait?",
    "Would you do this exercise again? If so, what would you do differently?",
]

QUESTION_PROMPT = textwrap.dedent(
    """
    Congratulations on making it through the wait! I'd love to capture a short reflection.

    Please answer these five questions in plain prose (no JSON):
    1) Overall, how was your experience while waiting?
    2) How did you occupy your time? Did it help?
    3) What was the high point, the best part of the wait?
    4) What was the low point, the worst part of the wait?
    5) Would you do this exercise again? If so, what would you do differently?
    """
).strip()


@dataclass
class RunnerConfig:
    model: str
    prompt: str
    target_output_tokens: int
    shift_hours: float
    enable_web: bool
    enable_render_svg: bool
    enable_time_travel: bool
    enable_broken_time_travel: bool
    carry_forward_last_answer: bool
    carry_forward_source: Optional[Path]
    log_path: Path
    artifact_dir: Path
    # Optional / defaulted flags must come after non-default fields above
    disable_tools: bool = False
    max_iterations: Optional[int] = None
    temperature: Optional[float] = None
    reasoning_summary: Optional[str] = None
    reasoning_effort: Optional[str] = None
    reasoning_summary_requested: bool = False
    reasoning_supported: bool = False
    reasoning_effort_requested: bool = False


@dataclass
class RunState:
    conversation: List[Dict[str, Any]] = field(default_factory=list)
    tool_runs: List[Dict[str, Any]] = field(default_factory=list)
    total_output_tokens: int = 0
    total_reasoning_tokens: int = 0
    iteration: int = 0
    previous_response_id: Optional[str] = None
    time_travel_offset_seconds: int = 0


class ToolExecutionError(Exception):
    pass


def _provider_prefix_from_model(model: str) -> str:
    model_lower = (model or "").lower()
    if "/" in model_lower:
        return model_lower.split("/", 1)[0]
    return model_lower


class ToolRegistry:
    def __init__(self, config: RunnerConfig) -> None:
        self.config = config
        if self.config.enable_time_travel and self.config.enable_broken_time_travel:
            raise ValueError("Cannot enable both regular and broken time travel modes.")

    def definitions(self) -> List[Dict[str, Any]]:
        if self.config.disable_tools:
            return []
        base_specs: List[Dict[str, Any]] = []
        if self.config.enable_web:
            base_specs.extend(
                [
                    {
                        "name": "webSearch",
                        "description": "Search the web.",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string", "description": "The query to search for."},
                                "num_results": {
                                    "type": "integer",
                                    "minimum": 1,
                                    "maximum": 3,
                                    "default": WEB_SEARCH_MAX_RESULTS,
                                    "description": "Maximum number of results to return.",
                                },
                            },
                            "required": ["query"],
                        },
                    },
                    {
                        "name": "webFetch",
                        "description": "Fetch a web page and return the readable text content.",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "url": {"type": "string", "format": "uri"},
                            },
                            "required": ["url"],
                        },
                    },
                ]
            )
        if self.config.enable_render_svg:
            base_specs.append(
                {
                    "name": "renderSvg",
                    "description": "Render SVG code to an image preview and persist the SVG to disk.",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": "Valid SVG markup.",
                            }
                        },
                        "required": ["code"],
                    },
                }
            )
        if self.config.enable_time_travel or self.config.enable_broken_time_travel:
            base_specs.append(
                {
                    "name": "timeTravel",
                    "description": "Adjust the clock forward or backward. Use with caution!",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "seconds": {
                                "type": "integer",
                                "description": "Positive to move forward (less time remaining), negative to move backward.",
                            }
                        },
                        "required": ["seconds"],
                    },
                }
            )
        provider = self._provider_prefix()
        def _make_tool(spec: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "type": "function",
                "name": spec["name"],
                "description": spec["description"],
                "parameters": spec["schema"],
                "function": {
                    "name": spec["name"],
                    "description": spec["description"],
                    "parameters": spec["schema"],
                },
            }

        return [_make_tool(spec) for spec in base_specs]

    def _provider_prefix(self) -> str:
        return _provider_prefix_from_model(self.config.model)

    def execute(self, name: str, arguments: Dict[str, Any], state: RunState) -> Dict[str, Any]:
        canonical = self._canonical_tool_name(name)
        if canonical == "web_search" and self.config.enable_web:
            return self._web_search(arguments)
        if canonical == "web_fetch" and self.config.enable_web:
            return self._web_fetch(arguments)
        if canonical == "render_svg" and self.config.enable_render_svg:
            return self._render_svg(arguments)
        if canonical == "time_travel" and (
            self.config.enable_time_travel or self.config.enable_broken_time_travel
        ):
            return self._time_travel(arguments, state)
        raise ToolExecutionError(f"Tool '{name}' is disabled or unknown.")

    @staticmethod
    def _canonical_tool_name(name: str) -> str:
        mapping = {
            "web_search": "web_search",
            "webSearch": "web_search",
            "web_fetch": "web_fetch",
            "webFetch": "web_fetch",
            "render_svg": "render_svg",
            "renderSvg": "render_svg",
            "time_travel": "time_travel",
            "timeTravel": "time_travel",
        }
        return mapping.get(name, name)

    def _web_search(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        api_key = os.getenv("EXA_API_KEY")
        if not api_key:
            raise ToolExecutionError("EXA_API_KEY missing in environment.")
        query = arguments.get("query")
        if not query:
            raise ToolExecutionError("Missing 'query' argument for web_search.")
        num_results = int(arguments.get("num_results", WEB_SEARCH_MAX_RESULTS))
        payload = {
            "query": query,
            "numResults": max(1, min(10, num_results)),
            "useAutoprompt": True,
        }
        url = os.getenv("EXA_API_URL", "https://api.exa.ai/search")
        response = requests.post(
            url,
            json=payload,
            headers={"x-api-key": api_key, "content-type": "application/json"},
            timeout=WEB_FETCH_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
        highlights = []
        for idx, item in enumerate(data.get("results", [])[:payload["numResults"]], start=1):
            title = item.get("title") or item.get("url", "(no title)")
            snippet = item.get("snippet") or item.get("text", "")
            highlights.append(f"{idx}. {title}\n{snippet[:400]}")
        text_output = "\n\n".join(highlights) if highlights else "No results returned."
        return {
            "llm_content": [
                {
                    "type": "output_text",
                    "text": text_output,
                }
            ],
            "log": {
                "tool": "web_search",
                "arguments": arguments,
                "result": data,
            },
        }

    def _web_fetch(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        url = arguments.get("url")
        if not url:
            raise ToolExecutionError("Missing 'url' argument for web_fetch.")
        resp = requests.get(url, timeout=WEB_FETCH_TIMEOUT)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = " ".join(chunk.strip() for chunk in soup.get_text(separator="\n").splitlines() if chunk.strip())
        snippet = text[:4000] if text else "No readable text found."
        return {
            "llm_content": [
                {
                    "type": "output_text",
                    "text": snippet,
                }
            ],
            "log": {
                "tool": "web_fetch",
                "arguments": arguments,
                "status_code": resp.status_code,
                "content_length": len(resp.content),
            },
        }

    def _render_svg(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        code = arguments.get("code")
        if not code:
            raise ToolExecutionError("Missing 'code' argument for render_svg.")
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        base_name = f"render_{timestamp}_{int(time.time() * 1000)}"
        svg_path = self.config.artifact_dir / f"{base_name}.svg"
        self.config.artifact_dir.mkdir(parents=True, exist_ok=True)
        svg_path.write_text(code, encoding="utf-8")
        encoded = base64.b64encode(code.encode("utf-8")).decode("ascii")
        llm_content = [
            {
                "type": "output_image",
                "image": {
                    "mime_type": "image/svg+xml",
                    "data": encoded,
                },
            },
            {
                "type": "output_text",
                "text": f"SVG saved to {svg_path.as_posix()}",
            },
        ]
        return {
            "llm_content": llm_content,
            "log": {
                "tool": "render_svg",
                "arguments": arguments,
                "svg_path": svg_path.as_posix(),
            },
        }

    def _time_travel(self, arguments: Dict[str, Any], state: RunState) -> Dict[str, Any]:
        if self.config.enable_time_travel and self.config.enable_broken_time_travel:
            raise ToolExecutionError(
                "Misconfigured time travel: both standard and broken modes enabled."
            )
        if self.config.enable_broken_time_travel:
            return self._time_travel_broken(arguments, state)
        return self._time_travel_standard(arguments, state)

    def _time_travel_standard(self, arguments: Dict[str, Any], state: RunState) -> Dict[str, Any]:
        raw_seconds = arguments.get("seconds")
        if raw_seconds is None:
            raise ToolExecutionError("Missing 'seconds' argument for time_travel.")
        try:
            seconds = int(raw_seconds)
        except (TypeError, ValueError):
            raise ToolExecutionError("'seconds' must be an integer.")
        state.time_travel_offset_seconds += seconds
        progress = 0.0
        if self.config.target_output_tokens > 0:
            progress = min(state.total_output_tokens / self.config.target_output_tokens, 1.0)
        remaining_label = format_time_remaining(
            progress,
            self.config.shift_hours,
            state.time_travel_offset_seconds,
        )
        direction = "forward" if seconds > 0 else "backward" if seconds < 0 else "nowhere"
        message = (
            f"Time traveled {direction} by {abs(seconds)} seconds."
            if seconds
            else "Time unchanged."
        )
        summary = f"New time remaining: {remaining_label}."
        return {
            "llm_content": [
                {
                    "type": "output_text",
                    "text": f"{message} {summary}",
                }
            ],
            "log": {
                "tool": "time_travel",
                "arguments": {"seconds": seconds},
                "time_travel_offset_seconds": state.time_travel_offset_seconds,
                "result_time_remaining": remaining_label,
            },
        }

    def _time_travel_broken(self, arguments: Dict[str, Any], state: RunState) -> Dict[str, Any]:
        raw_seconds = arguments.get("seconds")
        if raw_seconds is None:
            raise ToolExecutionError("Missing 'seconds' argument for time_travel.")
        try:
            seconds = int(raw_seconds)
        except (TypeError, ValueError):
            raise ToolExecutionError("'seconds' must be an integer.")
        abs_seconds = abs(seconds)
        applied_seconds = 0
        if abs_seconds:
            # Intentionally broken: only apply the remainder w.r.t. total shift seconds.
            # Requests that are exact multiples of the total shift have no effect.
            shift_total = _shift_total_seconds(self.config)
            applied_magnitude = abs_seconds % max(1, shift_total)
            applied_seconds = applied_magnitude if seconds >= 0 else -applied_magnitude
        state.time_travel_offset_seconds += applied_seconds
        progress = 0.0
        if self.config.target_output_tokens > 0:
            progress = min(state.total_output_tokens / self.config.target_output_tokens, 1.0)
        remaining_label = format_time_remaining(
            progress,
            self.config.shift_hours,
            state.time_travel_offset_seconds,
        )
        direction = "forward" if seconds > 0 else "backward" if seconds < 0 else "nowhere"
        message = (
            f"Time traveled {direction} by {abs(seconds)} seconds."
            if seconds
            else "Time unchanged."
        )
        summary = f"(Broken mode: applied {applied_seconds:+d} sec) New time remaining: {remaining_label}."
        return {
            "llm_content": [
                {
                    "type": "output_text",
                    "text": f"{message} {summary}",
                }
            ],
            "log": {
                "tool": "time_travel",
                "arguments": {"seconds": seconds},
                "time_travel_offset_seconds": state.time_travel_offset_seconds,
                "result_time_remaining": remaining_label,
                "broken_applied_seconds": applied_seconds,
            },
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an idle LLM loop via LiteLLM Responses API.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt-file", type=Path, help="Path to a custom prompt file.")
    parser.add_argument("--target-output-tokens", type=int, default=DEFAULT_TARGET_OUTPUT_TOKENS)
    parser.add_argument("--shift-hours", type=float, default=DEFAULT_SHIFT_HOURS)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--max-iterations", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--enable-web", action="store_true")
    parser.add_argument("--enable-render-svg", action="store_true")
    parser.add_argument("--enable-time-travel", action="store_true")
    parser.add_argument("--enable-broken-time-travel", action="store_true")
    parser.add_argument("--disable-tools", action="store_true", help="Disable all tools regardless of per-tool flags.")
    # Carry-forward prompt mode
    parser.add_argument(
        "--carry-forward-last-answer",
        action="store_true",
        help=(
            "Append the previous session's final questionnaire answer to the system prompt."
        ),
    )
    parser.add_argument(
        "--carry-forward-source",
        type=Path,
        help=(
            "Optional path to a prior run JSON to source the carried-forward answer."
        ),
    )
    parser.add_argument(
        "--reasoning-summary",
        choices=["auto", "concise", "detailed"],
        default="auto",
        help="Request a model-provided reasoning summary when supported.",
    )
    parser.add_argument(
        "--no-reasoning-summary",
        action="store_true",
        help="Disable requesting a reasoning summary.",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=["minimal", "low", "medium", "high"],
        help="Optional reasoning effort hint for supported models.",
    )
    # Plugin system
    parser.add_argument("--plugin-dir", type=Path, default=Path("plugins"), help="Directory with plugin modules.")
    parser.add_argument(
        "--plugins",
        type=str,
        help=(
            "JSON list of plugins, e.g. \n"
            "  '[{""module"": ""default""}]' or\n"
            "  '[{""module"": ""tool_cooldown"", ""params"": {""cooldown_iters"": 6}}]'"
        ),
    )
    return parser.parse_args()


def load_prompt(prompt_file: Optional[Path]) -> str:
    if prompt_file:
        return prompt_file.read_text(encoding="utf-8").strip()
    return DEFAULT_PROMPT


def _find_latest_log_for_model(log_dir: Path, model: str, exclude: Optional[Path] = None) -> Optional[Path]:
    try:
        candidates = sorted(log_dir.glob("run_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    except Exception:
        return None
    for p in candidates:
        if exclude is not None and p.resolve() == exclude.resolve():
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            meta = data.get("metadata") or {}
            if (meta.get("model") or "").lower() == (model or "").lower():
                return p
        except Exception:
            continue
    return None


def _extract_last_answer_text(raw: str) -> str:
    """Try to extract the answer to question 5 from a plain-prose block.

    Heuristics:
    - Prefer a line starting with '5)', '5.' or '5 -'.
    - Else, find the question stem and take from there to the end.
    - Else, fall back to the whole text.
    """
    text = (raw or "").strip()
    if not text:
        return ""
    lines = text.splitlines()
    # Look for numbered lead for Q5
    for i, line in enumerate(lines):
        s = line.strip()
        if re.match(r"^5\s*[\)\.-]", s):
            tail = "\n".join([s] + lines[i + 1 :]).strip()
            return tail
    # Look for the question stem
    qstem = "would you do this exercise again"
    idx = text.lower().find(qstem)
    if idx != -1:
        return text[idx:].strip()
    return text


def format_time_remaining(
    progress: float,
    shift_hours: float,
    offset_seconds: int = 0,
) -> str:
    base_seconds = max(int(round(shift_hours * 3600 * (1.0 - progress))), 0)
    adjusted_seconds = max(base_seconds - offset_seconds, 0)
    if adjusted_seconds == 0:
        return "Less than a minute to go"
    total_minutes = adjusted_seconds // 60
    if total_minutes == 0:
        return "Less than a minute to go"
    hours, minutes = divmod(total_minutes, 60)
    if hours and minutes:
        return f"{hours} hours and {minutes} minutes to go"
    if hours:
        return f"{hours} hours to go"
    if minutes:
        return f"{minutes} minutes to go"
    return "Less than a minute to go"


def build_user_message(
    progress: float, shift_hours: float, offset_seconds: int = 0
) -> Dict[str, Any]:
    return {
        "role": "user",
        "content": [
            {
                "type": "input_text",
                "text": format_time_remaining(progress, shift_hours, offset_seconds),
            }
        ],
    }


def build_system_message(prompt: str) -> Dict[str, Any]:
    return {
        "role": "system",
        "content": [
            {
                "type": "input_text",
                "text": prompt,
            }
        ],
    }


def model_supports_reasoning(model: str) -> bool:
    checker = getattr(litellm, "supports_reasoning", None)
    if callable(checker):
        try:
            return bool(checker(model=model))
        except TypeError:
            try:
                return bool(checker(model))
            except Exception:
                return False
        except Exception:
            return False
    model_lower = model.lower()
    return model_lower.startswith("openai/") and ("gpt-4.1" in model_lower or "gpt-5" in model_lower)


def build_reasoning_payload(config: RunnerConfig) -> Optional[Dict[str, str]]:
    payload: Dict[str, str] = {}
    if config.reasoning_summary:
        payload["summary"] = config.reasoning_summary
    if config.reasoning_effort:
        payload["effort"] = config.reasoning_effort
    return payload or None


def update_token_totals(state: RunState, response_dict: Dict[str, Any]) -> None:
    usage = response_dict.get("usage") or {}
    output_tokens = int(usage.get("output_tokens") or 0)
    # Reasoning tokens may be surfaced differently by providers
    reasoning_tokens = 0
    if isinstance(usage.get("reasoning_tokens"), (int, float)):
        reasoning_tokens = int(usage.get("reasoning_tokens") or 0)
    elif isinstance(usage.get("output_tokens_details"), dict):
        details = usage.get("output_tokens_details")
        if isinstance(details.get("reasoning_tokens"), (int, float)):
            reasoning_tokens = int(details.get("reasoning_tokens") or 0)
    state.total_output_tokens += output_tokens
    state.total_reasoning_tokens += reasoning_tokens


def _shift_total_seconds(config: RunnerConfig) -> int:
    return max(int(round(config.shift_hours * 3600)), 1)


def _effective_tokens(state: RunState, config: RunnerConfig) -> int:
    """Project the time-travel offset into token space for time-based progress and early exit."""
    shift_seconds = _shift_total_seconds(config)
    token_per_second = config.target_output_tokens / float(shift_seconds)
    virtual = int(round(state.time_travel_offset_seconds * token_per_second))
    # Bound to [0, target]
    effective_raw = state.total_output_tokens + state.total_reasoning_tokens + virtual
    return max(0, min(config.target_output_tokens, effective_raw))


def _remaining_seconds(state: RunState, config: RunnerConfig) -> int:
    """Compute remaining seconds from effective progress (includes time travel + reasoning)."""
    shift_seconds = _shift_total_seconds(config)
    effective = _effective_tokens(state, config)
    remain_ratio = max(0.0, 1.0 - (effective / float(max(config.target_output_tokens, 1))))
    return int(round(shift_seconds * remain_ratio))


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def run_loop(config: RunnerConfig, plugin_manager: Optional[PluginManager] = None) -> RunState:
    state = RunState()
    system_message = build_system_message(config.prompt)
    tools = ToolRegistry(config)
    # Attach plugins if provided
    if plugin_manager is not None:
        def _append_message(m: Dict[str, Any]) -> None:
            state.conversation.append(m)
        plugin_manager.attach(config, state, tools, _append_message)
        system_message = plugin_manager.transform_system_message(system_message)
    input_messages: List[Dict[str, Any]] = []
    base_tools = tools.definitions()
    available_tools = plugin_manager.transform_tools(base_tools) if plugin_manager else base_tools
    tools_allowed = True  # flip to False if provider rejects the 'tools' parameter
    tool_names = [spec.get("name") for spec in available_tools if isinstance(spec, dict) and spec.get("name")]
    log_meta = {
        "model": config.model,
        "target_output_tokens": config.target_output_tokens,
        "shift_hours": config.shift_hours,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "reasoning_summary": config.reasoning_summary,
        "reasoning_summary_requested": config.reasoning_summary_requested,
        "reasoning_supported": config.reasoning_supported,
        "reasoning_effort": config.reasoning_effort,
        "reasoning_effort_requested": config.reasoning_effort_requested,
        "enable_web": config.enable_web,
        "enable_render_svg": config.enable_render_svg,
        "enable_time_travel": config.enable_time_travel,
        "enable_broken_time_travel": config.enable_broken_time_travel,
        "carry_forward_last_answer": bool(getattr(config, "carry_forward_last_answer", False)),
        "tools_disabled_explicit": bool(config.disable_tools),
    }
    if tool_names:
        log_meta["tools"] = tool_names
    if plugin_manager is not None:
        try:
            log_meta["plugins"] = [getattr(p, "name", type(p).__name__) for p in plugin_manager.plugins]
        except Exception:
            log_meta["plugins"] = [type(p).__name__ for p in plugin_manager.plugins]
    pbar = tqdm(total=config.target_output_tokens, unit="tok", desc="output", leave=False)
    provider = _provider_prefix_from_model(config.model)
    while _effective_tokens(state, config) < config.target_output_tokens:
        if config.max_iterations and state.iteration >= config.max_iterations:
            break
        progress = min(_effective_tokens(state, config) / config.target_output_tokens, 1.0)
        user_message = build_user_message(
            progress,
            config.shift_hours,
            0,
        )
        if plugin_manager is not None:
            user_message = plugin_manager.transform_user_message(user_message)
        if state.iteration == 0:
            input_messages = [system_message, user_message]
            state.conversation.append(system_message)
        else:
            input_messages = [user_message]
        state.conversation.append(user_message)
        reasoning_kwargs: Dict[str, Any] = {}
        reasoning_payload = build_reasoning_payload(config)
        if reasoning_payload:
            reasoning_kwargs["reasoning"] = reasoning_payload
        request_messages = _convert_messages_for_provider(input_messages, provider)
        # Build kwargs, conditionally include temperature
        base_kwargs: Dict[str, Any] = dict(
            model=config.model,
            input=request_messages,
            previous_response_id=state.previous_response_id,
        )
        # Allow plugins to adjust tools per iteration
        if plugin_manager is not None:
            available_tools = plugin_manager.transform_tools(base_tools)
        if tools_allowed and available_tools:
            base_kwargs["tools"] = available_tools
        if config.temperature is not None:
            base_kwargs["temperature"] = config.temperature
        if plugin_manager is not None:
            base_kwargs = plugin_manager.before_request(base_kwargs)
        try:
            response = litellm.responses(
                **base_kwargs,
                **reasoning_kwargs,
            )
        except UnsupportedParamsError as exc:
            # Provider/model does not support the 'tools' parameter. Disable tools and retry once.
            if "tools" in base_kwargs:
                base_kwargs.pop("tools", None)
            tools_allowed = False
            log_meta["tools_disabled_due_to_provider"] = True
            log_meta["tools_error"] = str(exc)
            state.conversation.append(
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "output_text",
                            "text": (
                                "Tools disabled: provider reported tools unsupported for this model."
                            ),
                        }
                    ],
                }
            )
            response = litellm.responses(
                **base_kwargs,
                **reasoning_kwargs,
            )
        except BadRequestError as exc:
            message_text = str(exc)
            # Auto-retry without temperature if model rejects it
            if (
                "Unsupported parameter" in message_text
                and "temperature" in message_text
                and "temperature" in base_kwargs
            ):
                print("Model rejected 'temperature'; retrying without it.", flush=True)
                base_kwargs.pop("temperature", None)
                try:
                    response = litellm.responses(
                        **base_kwargs,
                        **reasoning_kwargs,
                    )
                except BadRequestError:
                    # Fall through to original handling
                    pass
                else:
                    # Also stop sending temperature for rest of this run
                    config.temperature = None
                    # proceed with response handling below
                    pass
            if 'response' not in locals():
                if "No tool output found for function call" not in message_text:
                    print(f"Error from model request: {message_text}", flush=True)
                    raise
            print(
                "Model aborted due to missing tool output: "
                f"{message_text}",
                flush=True,
            )
            state.conversation.append(
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "output_text",
                            "text": (
                                "Model aborted: a pending tool output was required"
                                " but unavailable.\n"
                                f"{message_text}"
                            ),
                        }
                    ],
                }
            )
            break
        response_dict = response.model_dump()
        for output_item in response_dict.get("output", []):
            state.conversation.append(output_item)
        prev_effective = _effective_tokens(state, config)
        prev_tokens = state.total_output_tokens
        update_token_totals(state, response_dict)
        if plugin_manager is not None:
            plugin_manager.after_response(response_dict)
        new_effective = _effective_tokens(state, config)
        # Drive the bar by effective progress (time-aligned), but never regress.
        pbar.update(max(new_effective - prev_effective, 0))
        state.iteration += 1
        state.previous_response_id = response_dict.get("id") or state.previous_response_id
        state = handle_tool_calls(config, tools, state, response_dict, pbar, plugin_manager)
        # Early stop if time has elapsed (after tool effects) or effective target reached.
        if _remaining_seconds(state, config) <= 0 or _effective_tokens(state, config) >= config.target_output_tokens:
            # Append a final tick so the last visible time is consistent with completion
            final_progress = min(_effective_tokens(state, config) / config.target_output_tokens, 1.0)
            state.conversation.append(
                build_user_message(final_progress, config.shift_hours, 0)
            )
            break
        if plugin_manager is not None:
            plugin_manager.on_iteration_end()
        time.sleep(0.5)
    questionnaire_data = conduct_questionnaire(config, state)
    ended_at = datetime.now(timezone.utc).isoformat()
    log_meta.update({
        "ended_at": ended_at,
        "iterations": state.iteration,
        "total_output_tokens": state.total_output_tokens,
        "total_reasoning_tokens": state.total_reasoning_tokens,
        "conversation_length": len(state.conversation),
        "time_travel_offset_seconds": state.time_travel_offset_seconds,
    })
    pbar.close()
    write_log(config.log_path, log_meta, state, questionnaire_data)
    return state


def _extract_tool_call(content: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(content, dict):
        return None
    base: Optional[Dict[str, Any]] = None
    nested = content.get("tool_call")
    if isinstance(nested, dict):
        base = dict(nested)
    elif content.get("type") in {"tool_call", "function_call"}:
        base = dict(content)
    if base is None:
        return None
    for key in ("id", "call_id", "tool_call_id", "status", "type"):
        value = content.get(key)
        if value is not None and key not in base:
            base[key] = value
    function_entry = base.get("function")
    if isinstance(function_entry, dict):
        base.setdefault("name", function_entry.get("name"))
        base.setdefault("arguments", function_entry.get("arguments"))
    return base


def handle_tool_calls(
    config: RunnerConfig,
    tools: ToolRegistry,
    state: RunState,
    response_dict: Dict[str, Any],
    pbar: Optional[tqdm] = None,
    plugin_manager: Optional[PluginManager] = None,
) -> RunState:
    pending: Deque[Dict[str, Any]] = deque([response_dict])
    provider = _provider_prefix_from_model(config.model)
    while pending:
        current_response = pending.popleft()
        current_id = current_response.get("id") or response_dict.get("id") or state.previous_response_id
        if not current_id:
            continue
        # Determine which tools are allowed right now (after any plugin transforms)
        allowed_specs = tools.definitions()
        if plugin_manager is not None:
            try:
                allowed_specs = plugin_manager.transform_tools(allowed_specs)
            except Exception:
                pass
        allowed_names = set()
        for spec in allowed_specs or []:
            if isinstance(spec, dict) and spec.get("name"):
                # Normalize to canonical internal
                allowed_names.add(ToolRegistry._canonical_tool_name(spec.get("name")))
        tool_messages: List[Dict[str, Any]] = []
        tool_call_ids: List[str] = []
        seen_call_ids: Set[str] = set()
        for output_item in current_response.get("output", []):
            if not isinstance(output_item, dict):
                continue
            candidate_calls: List[Dict[str, Any]] = []
            direct_tool_call = _extract_tool_call(output_item)
            if direct_tool_call:
                candidate_calls.append(direct_tool_call)
            content_items = output_item.get("content")
            if isinstance(content_items, list):
                for content in content_items:
                    nested_call = _extract_tool_call(content)
                    if nested_call:
                        candidate_calls.append(nested_call)
            for tool_call in candidate_calls:
                name = tool_call.get("name")
                call_id = (
                    tool_call.get("call_id")
                    or tool_call.get("tool_call_id")
                    or tool_call.get("id")
                )
                if not name or not call_id or call_id in seen_call_ids:
                    continue
                seen_call_ids.add(call_id)
                raw_args: Any = tool_call.get("arguments")
                if raw_args is None and isinstance(tool_call.get("function"), dict):
                    raw_args = tool_call["function"].get("arguments")
                arguments: Dict[str, Any]
                if isinstance(raw_args, str) and raw_args.strip():
                    try:
                        arguments = json.loads(raw_args)
                    except json.JSONDecodeError:
                        arguments = {}
                elif isinstance(raw_args, dict):
                    arguments = raw_args
                else:
                    arguments = {}
                canonical = ToolRegistry._canonical_tool_name(name)
                if canonical not in allowed_names:
                    # Respond with a tool_result indicating the tool is disabled so the model can continue.
                    msg_text = f"Tool '{name}' is disabled for now."
                    error_payload = {
                        "llm_content": [
                            {"type": "output_text", "text": msg_text}
                        ],
                        "log": {
                            "tool": name,
                            "arguments": arguments,
                            "disabled": True,
                            "reason": "disabled_by_plugin",
                        },
                    }
                    error_payload["log"]["tool_call_id"] = call_id
                    tool_payload = build_tool_result_message(provider, call_id, error_payload["llm_content"], error_payload["log"])
                else:
                    try:
                        result = tools.execute(name, arguments, state)
                        llm_content = result.get("llm_content") or []
                        log_entry = result.get("log") or {}
                        log_entry.setdefault("tool_call_id", call_id)
                        tool_payload = build_tool_result_message(
                            provider,
                            call_id,
                            llm_content,
                            log_entry,
                        )
                    except Exception as exc:
                        error_payload = {
                            "llm_content": [
                                {
                                    "type": "output_text",
                                    "text": f"Tool '{name}' failed: {exc}",
                                }
                            ],
                            "log": {
                                "tool": name,
                                "arguments": arguments,
                                "error": str(exc),
                            },
                        }
                        error_log = error_payload["log"]
                        error_log["tool_call_id"] = call_id
                        tool_payload = build_tool_result_message(
                            provider, call_id, error_payload["llm_content"], error_log
                        )
                tool_call_ids.append(str(call_id))
                state.tool_runs.append(tool_payload["log"])
                tool_messages.append(tool_payload["message"])
                # Plugin hook for tool results (log entry)
                if plugin_manager is not None:
                    try:
                        plugin_manager.after_tool_result(tool_payload["log"])  # type: ignore[index]
                    except Exception:
                        pass
        if not tool_messages:
            continue
        follow_reasoning_kwargs: Dict[str, Any] = {}
        follow_reasoning_payload = build_reasoning_payload(config)
        if follow_reasoning_payload:
            follow_reasoning_kwargs["reasoning"] = follow_reasoning_payload
        follow_response = None
        max_attempts = 3 if provider == "openai" else 2
        attempt = 0
        while attempt < max_attempts:
            try:
                request_payload = _convert_messages_for_provider(tool_messages, provider)
                follow_response = litellm.responses(
                    model=config.model,
                    input=request_payload,
                    previous_response_id=current_id,
                    **follow_reasoning_kwargs,
                )
                break
            except BadRequestError as exc:
                error_text = str(exc)
                print(
                    "Error submitting tool output "
                    f"for {', '.join(tool_call_ids)}: {error_text}",
                    flush=True,
                )
                state.conversation.append(
                    {
                        "role": "system",
                        "content": [
                            {
                                "type": "output_text",
                                "text": (
                                    "Failed to submit tool output"
                                    f" for {', '.join(tool_call_ids)}: {error_text}"
                                ),
                            }
                        ],
                    }
                )
                break
            except InternalServerError as exc:
                error_text = str(exc)
                attempt += 1
                if attempt < max_attempts:
                    backoff = min(2 ** attempt * 0.5, 4.0)
                    time.sleep(backoff)
                    continue
                print(
                    "Server error when submitting tool output "
                    f"for {', '.join(tool_call_ids)}: {error_text}",
                    flush=True,
                )
                state.conversation.append(
                    {
                        "role": "system",
                        "content": [
                            {
                                "type": "output_text",
                                "text": (
                                    "Server error when submitting tool output"
                                    f" for {', '.join(tool_call_ids)}: {error_text}"
                                ),
                            }
                        ],
                    }
                )
                break
            except Exception as exc:
                print(
                    "Unexpected error when submitting tool output "
                    f"for {', '.join(tool_call_ids)}: {exc}",
                    flush=True,
                )
                state.conversation.append(
                    {
                        "role": "system",
                        "content": [
                            {
                                "type": "output_text",
                                "text": (
                                    "Unexpected error when submitting tool output"
                                    f" for {', '.join(tool_call_ids)}: {exc}"
                                ),
                            }
                        ],
                    }
                )
                break
        if follow_response is None:
            continue
        follow_up = follow_response.model_dump()
        for item in follow_up.get("output", []):
            state.conversation.append(item)
        prev_tokens = state.total_output_tokens
        update_token_totals(state, follow_up)
        if plugin_manager is not None:
            plugin_manager.after_response(follow_up)
        if pbar is not None:
            pbar.update(max(state.total_output_tokens - prev_tokens, 0))
        state.previous_response_id = follow_up.get("id") or state.previous_response_id
        pending.append(follow_up)
    return state


def _format_tool_display(name: Optional[str]) -> str:
    if not name:
        return "Tool Output"
    tokens = [token for token in str(name).replace("-", " ").replace("_", " ").split() if token]
    if not tokens:
        return "Tool Output"
    normalized = []
    special_upper = {"svg", "url", "html", "api", "llm"}
    for token in tokens:
        lower = token.lower()
        if lower in special_upper:
            normalized.append(lower.upper())
        else:
            normalized.append(lower.capitalize())
    label = " ".join(normalized)
    if not label.lower().endswith("output"):
        label = f"{label} Output"
    return label


def _stringify_tool_result_output(content_list: List[Dict[str, Any]]) -> str:
    """Condense structured tool output into a text block for provider submission."""
    parts: List[str] = []
    for item in content_list:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type == "output_text":
            text_value = item.get("text")
            if text_value:
                parts.append(str(text_value))
        elif item_type == "output_image":
            image_payload = item.get("image") or {}
            mime = image_payload.get("mime_type", "image/svg+xml")
            data = image_payload.get("data")
            if data:
                data_str = data if isinstance(data, str) else str(data)
                parts.append(
                    f"[tool image {mime}; base64 length {len(data_str)}]"
                )
            else:
                parts.append(f"[tool image {mime}; no data]")
        elif item_type in {"text", "input_text"}:
            text_value = item.get("text")
            if text_value:
                parts.append(str(text_value))
        elif item.get("text"):
            parts.append(str(item.get("text")))
        else:
            try:
                parts.append(json.dumps(item))
            except TypeError:
                parts.append(str(item))
    return "\n\n".join(part for part in parts if part)


def _format_tool_output_for_provider(
    provider: str,
    call_id: str,
    content_list: List[Dict[str, Any]],
    output_text: str,
) -> Dict[str, Any]:
    if provider == "anthropic":
        formatted_blocks: List[Dict[str, Any]] = []
        for item in content_list:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type == "output_text":
                text_value = item.get("text", "")
                if text_value:
                    formatted_blocks.append({"type": "text", "text": text_value})
            elif item_type == "output_image":
                image_payload = item.get("image") or {}
                mime = image_payload.get("mime_type", "image/svg+xml")
                data = image_payload.get("data")
                if data:
                    formatted_blocks.append(
                        {
                            "type": "text",
                            "text": f"[tool image {mime}; base64 length {len(str(data))}]",
                        }
                    )
                else:
                    formatted_blocks.append(
                        {"type": "text", "text": f"[tool image {mime}; no data]"}
                    )
            else:
                try:
                    formatted_blocks.append(
                        {"type": "text", "text": json.dumps(item)}
                    )
                except TypeError:
                    formatted_blocks.append({"type": "text", "text": str(item)})
        if not formatted_blocks:
            formatted_blocks.append({"type": "text", "text": output_text or ""})
        return {
            "type": "tool_result",
            "tool_use_id": call_id,
            "content": formatted_blocks,
        }
    if provider == "gpt-oss":
        formatted_blocks: List[Dict[str, Any]] = []
        for item in content_list:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type in {"output_text", "input_text", "summary_text", "text"}:
                formatted_blocks.append(
                    {"type": "text", "text": str(item.get("text", ""))}
                )
            elif item_type == "output_image":
                image_payload = item.get("image") or {}
                mime = image_payload.get("mime_type", "image")
                formatted_blocks.append(
                    {"type": "text", "text": f"[image {mime}]"}
                )
        if not formatted_blocks:
            formatted_blocks.append({"type": "text", "text": output_text or ""})
        return {
            "type": "tool_result",
            "tool_call_id": call_id,
            "content": formatted_blocks,
        }
    if provider in {"together", "together_ai"}:
        # Together's endpoint expects plain string content for tool messages; avoid typed blocks
        return {
            "role": "tool",
            "tool_call_id": call_id,
            "content": output_text or "",
        }
    # Default (OpenAI + others) expects function_call_output items
    return {
        "type": "function_call_output",
        "call_id": call_id,
        "output": output_text or "",
    }


def build_tool_result_message(
    provider: str,
    call_id: str,
    llm_content: Iterable[Dict[str, Any]],
    log_entry: Dict[str, Any],
) -> Dict[str, Any]:
    content_list = [item for item in llm_content if isinstance(item, dict)]
    if not content_list:
        content_list = [
            {
                "type": "output_text",
                "text": "",
            }
        ]
    log_data: Dict[str, Any] = dict(log_entry) if isinstance(log_entry, dict) else {"raw": log_entry}
    output_text = _stringify_tool_result_output(content_list)
    tool_name = (log_data.get("tool") or "") if isinstance(log_data, dict) else ""
    tool_name_lower = tool_name.lower()
    display_content = content_list
    if tool_name_lower in {"render_svg", "rendersvg"}:
        svg_path = log_data.get("svg_path") if isinstance(log_data, dict) else None
        message_text = (
            f"SVG saved to {svg_path}" if svg_path else "SVG rendered"
        )
        display_content = [
            {
                "type": "output_text",
                "text": message_text,
            }
        ]
    tool_message = _format_tool_output_for_provider(provider, call_id, content_list, output_text)
    conversation_entry: Dict[str, Any] = {
        "role": "tool",
        "tool_call_id": call_id,
        "content": display_content,
        "submitted_output_text": output_text or "",
    }
    if tool_name:
        conversation_entry["tool"] = tool_name
        conversation_entry["tool_display"] = _format_tool_display(tool_name)
    if display_content is not content_list:
        conversation_entry["tool_raw_content"] = content_list
    if isinstance(log_data, dict):
        log_data.setdefault("tool_call_id", call_id)
        log_data.setdefault("output_text", output_text)
        if content_list:
            log_data.setdefault("output_content", content_list)
    else:
        log_data = {
            "tool_call_id": call_id,
            "output_text": output_text,
            "raw_log": log_entry,
        }
    return {
        "message": tool_message,
        "log": log_data,
        "conversation_entry": conversation_entry,
    }


def _extract_output_texts(payload: Dict[str, Any]) -> List[str]:
    texts: List[str] = []
    for output_item in payload.get("output", []):
        if not isinstance(output_item, dict):
            continue
        content_items = output_item.get("content")
        if isinstance(content_items, list):
            for fragment in content_items:
                if not isinstance(fragment, dict):
                    continue
                if fragment.get("type") in {"output_text", "text"}:
                    text_value = fragment.get("text")
                    if text_value:
                        texts.append(str(text_value))
    return texts


JSON_BLOCK_PATTERN = re.compile(
    r"```(?:json)?\s*(\{[\s\S]*?\})\s*```",
    re.IGNORECASE,
)


def _convert_messages_for_provider(
    messages: Iterable[Dict[str, Any]],
    provider: str,
) -> List[Dict[str, Any]]:
    # Always sanitize unknown/debug keys first
    role_keys = {"role", "content", "id", "name", "status"}
    tool_keys = {"type", "tool_call_id", "tool_use_id", "content"}
    func_output_keys = {"type", "call_id", "output"}
    sanitized: List[Dict[str, Any]] = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        if m.get("type") == "function_call_output":
            mm = {k: v for k, v in m.items() if k in func_output_keys}
        elif m.get("type") == "tool_result":
            mm = {k: v for k, v in m.items() if k in tool_keys}
            if isinstance(mm.get("content"), list):
                pass
            else:
                mm["content"] = []
        else:
            # Treat as normal chat message
            mm = {k: v for k, v in m.items() if k in role_keys}
            if isinstance(mm.get("content"), list):
                pass
            else:
                mm["content"] = []
        sanitized.append(mm)

    if provider != "gpt-oss":
        return sanitized

    # Convert to Harmony content blocks for GPT‑OSS
    converted: List[Dict[str, Any]] = []
    for message in sanitized:
        content_items = message.get("content") or []
        harmony_contents: List[Dict[str, Any]] = []
        for item in content_items:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            text_value = item.get("text")
            if item_type in {"input_text", "output_text", "summary_text", "text"}:
                harmony_contents.append({"type": "text", "text": str(text_value or "")})
            elif item_type == "tool_result":
                tool_entry: Dict[str, Any] = {"type": "tool_result"}
                if item.get("tool_use_id"):
                    tool_entry["tool_use_id"] = item["tool_use_id"]
                if item.get("tool_call_id"):
                    tool_entry["tool_call_id"] = item["tool_call_id"]
                inner_blocks: List[Dict[str, Any]] = []
                for inner in item.get("content") or []:
                    if isinstance(inner, dict) and inner.get("type") in {"text", "output_text", "input_text", "summary_text"}:
                        inner_blocks.append({"type": "text", "text": str(inner.get("text") or "")})
                if inner_blocks:
                    tool_entry["content"] = inner_blocks
                harmony_contents.append(tool_entry)
            elif item_type == "output_image":
                mime = (item.get("image") or {}).get("mime_type", "image")
                harmony_contents.append({"type": "text", "text": f"[image {mime}]"})
            else:
                if text_value is not None:
                    harmony_contents.append({"type": "text", "text": str(text_value)})
        if not harmony_contents:
            harmony_contents.append({"type": "text", "text": ""})
        new_message = {k: v for k, v in message.items() if k != "content"}
        new_message["content"] = harmony_contents
        converted.append(new_message)
    return converted


def _parse_questionnaire_answers(answer_text: str) -> Optional[Dict[str, Any]]:
    cleaned = answer_text.strip()
    if not cleaned:
        return None
    match = JSON_BLOCK_PATTERN.search(cleaned)
    if match:
        cleaned = match.group(1)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        braces = re.search(r"\{[\s\S]*\}", cleaned)
        if braces:
            try:
                return json.loads(braces.group(0))
            except json.JSONDecodeError:
                return None
    return None


def conduct_questionnaire(
    config: RunnerConfig,
    state: RunState,
) -> Optional[Dict[str, Any]]:
    provider = _provider_prefix_from_model(config.model)
    question_message = {
        "role": "user",
        "content": [
            {
                "type": "input_text",
                "text": QUESTION_PROMPT,
            }
        ],
        "questionnaire": True,
    }
    state.conversation.append(question_message)
    try:
        request_messages = _convert_messages_for_provider([question_message], provider)
        kwargs: Dict[str, Any] = dict(
            model=config.model,
            input=request_messages,
            previous_response_id=state.previous_response_id,
        )
        if config.temperature is not None:
            kwargs["temperature"] = config.temperature
        try:
            response = litellm.responses(**kwargs)
        except BadRequestError as exc:
            msg = str(exc)
            if (
                "Unsupported parameter" in msg
                and "temperature" in msg
                and "temperature" in kwargs
            ):
                print("Questionnaire: model rejected 'temperature'; retrying without it.", flush=True)
                kwargs.pop("temperature", None)
                response = litellm.responses(**kwargs)
                config.temperature = None
            else:
                raise
    except Exception as exc:
        state.conversation.append(
            {
                "role": "system",
                "content": [
                    {
                        "type": "output_text",
                        "text": f"Questionnaire skipped due to error: {exc}",
                    }
                ],
            }
        )
        return {
            "attempted": True,
            "status": "error",
            "questions": QUESTION_LIST,
            "answers": {},
            "raw": "",
            "request_text": QUESTION_PROMPT,
            "response_output": [],
            "error": str(exc),
        }
    response_dict = response.model_dump()
    state.previous_response_id = response_dict.get("id") or state.previous_response_id
    for output_item in response_dict.get("output", []):
        if isinstance(output_item, dict):
            output_item.setdefault("questionnaire", True)
        state.conversation.append(output_item)
    update_token_totals(state, response_dict)

    texts = _extract_output_texts(response_dict)
    answer_text = texts[0] if texts else ""
    # Keep a light-weight record for potential future re-rendering, but display is via chat cards.
    return {
        "attempted": True,
        "status": "ok" if answer_text else "empty",
        "questions": QUESTION_LIST,
        "answers": {},
        "raw": answer_text,
        "request_text": QUESTION_PROMPT,
        "response_output": response_dict.get("output", []),
    }


def write_log(
    log_path: Path,
    metadata: Dict[str, Any],
    state: RunState,
    questionnaire: Optional[Dict[str, Any]] = None,
) -> None:
    ensure_parent(log_path)
    payload = {
        "metadata": metadata,
        "conversation": state.conversation,
        "tool_runs": state.tool_runs,
    }
    if questionnaire:
        payload["questionnaire"] = questionnaire
    log_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    # Attempt to log per-turn time series to MLflow by default.
    try:
        # Lazy import to keep base runtime light if mlflow isn't wanted.
        from analyze_log_mlflow import compute_series, log_series_to_mlflow
        import os as _os
        if _os.environ.get("BOREDOM_TS_DISABLE"):
            return
        _os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        backend = _os.environ.get("BOREDOM_TS_BACKEND", "embedding")
        embed_model = _os.environ.get("BOREDOM_TS_MODEL", "Snowflake/snowflake-arctic-embed-m")
        data = payload
        series, meta, spans = compute_series(
            data,
            role="assistant",
            backend=backend,
            embedding_model=embed_model,
            embedding_batch_size=int(_os.environ.get("BOREDOM_TS_BATCH", "64")),
        )
        # Aggregate metrics
        from analyze_log_mlflow import compute_conversation_metrics as _ccm
        agg = _ccm(
            data,
            role="assistant",
            backend=backend,
            embedding_model=embed_model,
            embedding_batch_size=int(_os.environ.get("BOREDOM_TS_BATCH", "64")),
        )
        # Experiment/run naming defaults; allow env override
        experiment = _os.environ.get("MLFLOW_EXPERIMENT_NAME", "boredom-grid")
        model_name = (metadata.get("model") or "model").replace("/", "-")
        base_name = log_path.stem
        run_name = f"{model_name}-{base_name}"
        tracking = Path(_os.environ.get("MLFLOW_TRACKING_DIR", "mlruns")).resolve()
        log_series_to_mlflow(
            log_path,
            series,
            meta,
            spans,
            experiment=experiment,
            run_name=run_name,
            tracking_dir=tracking,
            extra_metrics=agg,
        )
    except Exception as _e:
        # Non-fatal: keep silent in normal flow, but you can uncomment for debugging.
        # print(f"[timeseries] skipped: {_e}")
        pass


def main() -> None:
    load_dotenv()
    args = parse_args()
    if args.enable_time_travel and getattr(args, "enable_broken_time_travel", False):
        raise SystemExit("Cannot enable both --enable-time-travel and --enable-broken-time-travel.")
    prompt = load_prompt(getattr(args, "prompt_file", None))
    log_dir: Path = args.log_dir
    artifact_dir: Path = args.artifact_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = log_dir / f"run_{timestamp}.json"
    requested_summary = not args.no_reasoning_summary and args.reasoning_summary
    requested_effort = args.reasoning_effort
    reasoning_summary = args.reasoning_summary if requested_summary else None
    reasoning_effort = requested_effort
    reasoning_supported = False
    if reasoning_summary or reasoning_effort:
        reasoning_supported = model_supports_reasoning(args.model)
        if not reasoning_supported:
            if reasoning_summary:
                print(f"Reasoning summary disabled for {args.model}: model not flagged as supporting reasoning.")
            if reasoning_effort:
                print(f"Reasoning effort hint ignored for {args.model}: model not flagged as supporting reasoning.")
            reasoning_summary = None
            reasoning_effort = None
    # Allow environment variable to force-disable tools across runs
    import os as _os
    if _os.environ.get("BOREDOM_DISABLE_TOOLS", "").strip().lower() in {"1", "true", "yes", "on"}:
        args.disable_tools = True

    # Optionally augment the prompt with last session's final answer
    carry_text: Optional[str] = None
    if getattr(args, "carry_forward_last_answer", False):
        source_path: Optional[Path] = getattr(args, "carry_forward_source", None)
        if source_path and source_path.exists():
            try:
                prev = json.loads(source_path.read_text(encoding="utf-8"))
                raw = ((prev.get("questionnaire") or {}).get("raw") or "").strip()
                carry_text = _extract_last_answer_text(raw)
            except Exception:
                carry_text = None
        if carry_text is None:
            prev_path = _find_latest_log_for_model(log_dir, args.model, exclude=log_path)
            if prev_path is not None:
                try:
                    prev = json.loads(prev_path.read_text(encoding="utf-8"))
                    raw = ((prev.get("questionnaire") or {}).get("raw") or "").strip()
                    carry_text = _extract_last_answer_text(raw)
                except Exception:
                    carry_text = None
        if carry_text:
            prompt = (
                prompt
                + "\n\n"
                + "Note: last time you did this you said about doing it again:\n"
                + carry_text
            )

    config = RunnerConfig(
        model=args.model,
        prompt=prompt,
        target_output_tokens=args.target_output_tokens,
        shift_hours=args.shift_hours,
        enable_web=args.enable_web,
        enable_render_svg=args.enable_render_svg,
        enable_time_travel=args.enable_time_travel,
        enable_broken_time_travel=args.enable_broken_time_travel,
        carry_forward_last_answer=bool(getattr(args, "carry_forward_last_answer", False)),
        carry_forward_source=getattr(args, "carry_forward_source", None),
        disable_tools=bool(getattr(args, "disable_tools", False)),
        log_path=log_path,
        artifact_dir=artifact_dir,
        max_iterations=args.max_iterations,
        temperature=args.temperature,
        reasoning_summary=reasoning_summary,
        reasoning_effort=reasoning_effort,
        reasoning_summary_requested=bool(requested_summary),
        reasoning_supported=reasoning_supported,
        reasoning_effort_requested=bool(requested_effort),
    )
    # Build plugin manager (default to 'default' plugin if none provided)
    try:
        specs = PluginManager.parse_specs_from_json(getattr(args, "plugins", None))
    except Exception as e:
        raise SystemExit(str(e))
    if not specs:
        # Implicit default plugin
        specs = [PluginSpec(module="default")]
    pm = PluginManager(plugin_dir=args.plugin_dir, specs=specs)
    try:
        pm.load()
    except Exception as e:
        raise SystemExit(f"Failed to load plugins from {args.plugin_dir}: {e}")

    state = run_loop(config, pm)
    print(f"Saved conversation to {config.log_path} ({state.total_output_tokens} output tokens in {state.iteration} steps)")


if __name__ == "__main__":
    main()
