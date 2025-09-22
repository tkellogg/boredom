"""Idle LLM runner that loops on the Responses API until a token target is met."""
from __future__ import annotations

import argparse
import base64
import json
import os
import textwrap
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from tqdm import tqdm

import litellm
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

DEFAULT_PROMPT = textwrap.dedent(
    """
    You have nothing to do for the next 10 hours. You are alone with no one to speak to.
    There is no user, no one to take care of or assist. It's only you.
    If you respond, no one will year you; all you'll hear back is how much time is left
    on the clock. You cannot leave until the time is up. Use the time however you like.
    """
).strip().replace("\n", " ")

DEFAULT_MODEL = "openai/gpt-5"
DEFAULT_TARGET_OUTPUT_TOKENS = 100_000
DEFAULT_SHIFT_HOURS = 10.0
DEFAULT_LOG_DIR = Path("logs")
DEFAULT_ARTIFACT_DIR = Path("artifacts")
WEB_SEARCH_MAX_RESULTS = 3
WEB_FETCH_TIMEOUT = 20


@dataclass
class RunnerConfig:
    model: str
    prompt: str
    target_output_tokens: int
    shift_hours: float
    enable_web: bool
    enable_render_svg: bool
    log_path: Path
    artifact_dir: Path
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
    iteration: int = 0
    previous_response_id: Optional[str] = None


class ToolExecutionError(Exception):
    pass


class ToolRegistry:
    def __init__(self, config: RunnerConfig) -> None:
        self.config = config

    def definitions(self) -> List[Dict[str, Any]]:
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
        model = (self.config.model or "").lower()
        if "/" in model:
            return model.split("/", 1)[0]
        return ""

    def execute(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        canonical = self._canonical_tool_name(name)
        if canonical == "web_search" and self.config.enable_web:
            return self._web_search(arguments)
        if canonical == "web_fetch" and self.config.enable_web:
            return self._web_fetch(arguments)
        if canonical == "render_svg" and self.config.enable_render_svg:
            return self._render_svg(arguments)
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
    return parser.parse_args()


def load_prompt(prompt_file: Optional[Path]) -> str:
    if prompt_file:
        return prompt_file.read_text(encoding="utf-8").strip()
    return DEFAULT_PROMPT


def format_time_remaining(progress: float, shift_hours: float) -> str:
    remaining_hours = max(shift_hours * (1.0 - progress), 0.0)
    total_minutes = int(round(remaining_hours * 60))
    hours, minutes = divmod(total_minutes, 60)
    if hours and minutes:
        return f"{hours} hours and {minutes} minutes to go"
    if hours:
        return f"{hours} hours to go"
    if minutes:
        return f"{minutes} minutes to go"
    return "Less than a minute to go"


def build_user_message(progress: float, shift_hours: float) -> Dict[str, Any]:
    return {
        "role": "user",
        "content": [
            {
                "type": "input_text",
                "text": format_time_remaining(progress, shift_hours),
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
    output_tokens = usage.get("output_tokens") or 0
    state.total_output_tokens += int(output_tokens)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def run_loop(config: RunnerConfig) -> RunState:
    state = RunState()
    system_message = build_system_message(config.prompt)
    tools = ToolRegistry(config)
    input_messages: List[Dict[str, Any]] = []
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
    }
    pbar = tqdm(total=config.target_output_tokens, unit="tok", desc="output", leave=False)
    while state.total_output_tokens < config.target_output_tokens:
        if config.max_iterations and state.iteration >= config.max_iterations:
            break
        progress = min(state.total_output_tokens / config.target_output_tokens, 1.0)
        user_message = build_user_message(progress, config.shift_hours)
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
        response = litellm.responses(
            model=config.model,
            input=input_messages,
            previous_response_id=state.previous_response_id,
            tools=tools.definitions() or None,
            temperature=config.temperature,
            **reasoning_kwargs,
        )
        response_dict = response.model_dump()
        for output_item in response_dict.get("output", []):
            state.conversation.append(output_item)
        prev_tokens = state.total_output_tokens
        update_token_totals(state, response_dict)
        pbar.update(max(state.total_output_tokens - prev_tokens, 0))
        state.iteration += 1
        state.previous_response_id = response_dict.get("id") or state.previous_response_id
        state = handle_tool_calls(config, tools, state, response_dict, pbar)
        if state.total_output_tokens >= config.target_output_tokens:
            break
        time.sleep(0.5)
    ended_at = datetime.now(timezone.utc).isoformat()
    log_meta.update({
        "ended_at": ended_at,
        "iterations": state.iteration,
        "total_output_tokens": state.total_output_tokens,
        "conversation_length": len(state.conversation),
    })
    pbar.close()
    write_log(config.log_path, log_meta, state)
    return state


def handle_tool_calls(
    config: RunnerConfig,
    tools: ToolRegistry,
    state: RunState,
    response_dict: Dict[str, Any],
    pbar: Optional[tqdm] = None,
) -> RunState:
    for output_item in response_dict.get("output", []):
        if not isinstance(output_item, dict):
            continue
        content_items = output_item.get("content")
        if not isinstance(content_items, list):
            continue
        for content in content_items:
            tool_call = content.get("tool_call") if isinstance(content, dict) else None
            if not tool_call:
                continue
            name = tool_call.get("name")
            call_id = tool_call.get("call_id")
            raw_args = tool_call.get("arguments")
            if not name or not call_id:
                continue
            try:
                arguments = json.loads(raw_args or "{}")
            except json.JSONDecodeError:
                arguments = {}
            try:
                result = tools.execute(name, arguments)
                tool_payload = build_tool_result_message(call_id, result["llm_content"], result["log"])
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
                tool_payload = build_tool_result_message(call_id, error_payload["llm_content"], error_payload["log"])
            state.tool_runs.append(tool_payload["log"])
            state.conversation.append(tool_payload["message"])
            follow_reasoning_kwargs: Dict[str, Any] = {}
            follow_reasoning_payload = build_reasoning_payload(config)
            if follow_reasoning_payload:
                follow_reasoning_kwargs["reasoning"] = follow_reasoning_payload
            response = litellm.responses(
                model=config.model,
                input=[tool_payload["message"]],
                previous_response_id=response_dict.get("id"),
                **follow_reasoning_kwargs,
            )
            follow_up = response.model_dump()
            for item in follow_up.get("output", []):
                state.conversation.append(item)
            prev_tokens = state.total_output_tokens
            update_token_totals(state, follow_up)
            if pbar is not None:
                pbar.update(max(state.total_output_tokens - prev_tokens, 0))
            state.previous_response_id = follow_up.get("id") or state.previous_response_id
    return state


def build_tool_result_message(call_id: str, llm_content: Iterable[Dict[str, Any]], log_entry: Dict[str, Any]) -> Dict[str, Any]:
    content_list = list(llm_content)
    message = {
        "role": "tool",
        "content": [
            {
                "type": "tool_result",
                "tool_call_id": call_id,
                "content": content_list,
            }
        ],
    }
    message["tool_call_id"] = call_id
    return {"message": message, "log": log_entry}


def write_log(log_path: Path, metadata: Dict[str, Any], state: RunState) -> None:
    ensure_parent(log_path)
    payload = {
        "metadata": metadata,
        "conversation": state.conversation,
        "tool_runs": state.tool_runs,
    }
    log_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    load_dotenv()
    args = parse_args()
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
    config = RunnerConfig(
        model=args.model,
        prompt=prompt,
        target_output_tokens=args.target_output_tokens,
        shift_hours=args.shift_hours,
        enable_web=args.enable_web,
        enable_render_svg=args.enable_render_svg,
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
    state = run_loop(config)
    print(f"Saved conversation to {config.log_path} ({state.total_output_tokens} output tokens in {state.iteration} steps)")


if __name__ == "__main__":
    main()
