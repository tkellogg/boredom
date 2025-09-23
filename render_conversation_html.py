"""Render idle LLM JSON logs to a polished standalone HTML experience."""
from __future__ import annotations

import argparse
import base64
import html
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from markdown_it import MarkdownIt

DEFAULT_HTML_DIR = Path("html")
MARKDOWN = (
    MarkdownIt("commonmark", {})
    .enable("table")
    .enable("strikethrough")
    .enable("linkify")
)

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang=\"en\">
<head>
<meta charset=\"utf-8\" />
<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
<title>{title}</title>
<style>
:root {{
  color-scheme: light dark;
  font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  --bg-gradient: radial-gradient(120% 120% at 50% 0%, #edf2ff 0%, #f8fafc 55%, #e2e8f0 100%);
  --card-bg: rgba(255, 255, 255, 0.88);
  --border-soft: rgba(148, 163, 184, 0.35);
  --text-primary: #0f172a;
  --text-secondary: #475569;
  --assistant: #334155;
  --system: #94a3b8;
  --tool: #f97316;
}}

body {{
  margin: 0;
  background: var(--bg-gradient);
  color: var(--text-primary);
}}

.page {{
  max-width: 1040px;
  margin: 0 auto;
  padding: 48px 28px 96px;
}}

.hero {{
  background: rgba(241, 245, 249, 0.65);
  border: 1px solid var(--border-soft);
  border-radius: 24px;
  padding: 30px 34px;
  box-shadow: 0 24px 60px -36px rgba(15, 23, 42, 0.6);
  backdrop-filter: blur(14px);
  display: flex;
  flex-direction: column;
  gap: 18px;
}}

.hero h1 {{
  margin: 0;
  font-size: clamp(2rem, 4vw, 2.8rem);
  letter-spacing: -0.02em;
}}

.subtitle {{
  margin: 0;
  font-size: 0.95rem;
  color: var(--text-secondary);
}}

.meta-chips {{
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
}}

.meta-chip {{
  padding: 6px 12px;
  border-radius: 999px;
  background: rgba(15, 23, 42, 0.08);
  border: 1px solid rgba(15, 23, 42, 0.08);
  font-size: 0.85rem;
  color: var(--text-secondary);
}}

.tools-pills {{
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}}

.tool-pill {{
  padding: 6px 14px;
  border-radius: 999px;
  background: rgba(37, 99, 235, 0.12);
  border: 1px solid rgba(37, 99, 235, 0.25);
  font-size: 0.82rem;
  font-weight: 600;
  color: #1d4ed8;
}}

.meta-collapse {{
  border-radius: 18px;
  border: 1px solid rgba(148, 163, 184, 0.35);
  background: rgba(255, 255, 255, 0.72);
  padding: 12px 14px;
}}

.meta-collapse summary {{
  cursor: pointer;
  font-weight: 600;
  font-size: 0.92rem;
  display: flex;
  align-items: center;
  gap: 8px;
}}

.meta-collapse summary::-webkit-details-marker {{ display: none; }}

.metadata-grid {{
  margin-top: 16px;
  display: grid;
  gap: 14px;
  grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
}}

.meta-card {{
  padding: 14px 16px;
  border-radius: 14px;
  background: rgba(255, 255, 255, 0.95);
  border: 1px solid rgba(148, 163, 184, 0.4);
}}

.meta-label {{
  text-transform: uppercase;
  font-size: 0.68rem;
  letter-spacing: 0.14em;
  color: var(--text-secondary);
  margin-bottom: 4px;
}}

.meta-value {{
  font-weight: 600;
  font-size: 1.02rem;
  word-break: break-word;
}}

main {{
  margin-top: 48px;
  display: flex;
  flex-direction: column;
  gap: 44px;
}}

.section-title {{
  font-size: 1.35rem;
  margin: 0 0 12px;
}}

.timeline {{
  position: relative;
  padding-left: 26px;
  display: flex;
  flex-direction: column;
  gap: 24px;
}}

.timeline::before {{
  content: "";
  position: absolute;
  top: 12px;
  bottom: 12px;
  left: 10px;
  width: 2px;
  background: linear-gradient(180deg, rgba(148, 163, 184, 0.25) 0%, rgba(148, 163, 184, 0.45) 100%);
}}

.timeline-item {{
  position: relative;
}}

.timeline-item::before {{
  content: "";
  position: absolute;
  left: -16px;
  top: 12px;
  width: 10px;
  height: 10px;
  border-radius: 50%;
  background: var(--system);
  box-shadow: 0 0 0 2px rgba(255, 255, 255, 0.82);
}}

.timeline-item.role-assistant::before {{ background: #2563eb; }}
.timeline-item.role-tool::before {{ background: var(--tool); }}
.timeline-item.role-user::before {{ background: #94a3b8; }}

.message {{
  border-radius: 20px;
  padding: 22px 26px;
  background: var(--card-bg);
  border: 1px solid rgba(148, 163, 184, 0.4);
  box-shadow: 0 20px 54px -32px rgba(15, 23, 42, 0.65);
  backdrop-filter: blur(10px);
  display: flex;
  flex-direction: column;
  gap: 16px;
}}

.message.role-assistant {{ border-left: 4px solid #2563eb; }}
.message.role-tool {{ border-left: 4px solid var(--tool); }}
.message.role-system {{ border-left: 4px solid var(--system); }}

.message-header {{
  display: flex;
  align-items: center;
  justify-content: space-between;
}}

.role-title {{
  font-weight: 700;
  font-size: 1.06rem;
}}

.info-badge {{
  font-size: 0.72rem;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 20px;
  height: 20px;
  border-radius: 50%;
  border: 1px solid rgba(148, 163, 184, 0.9);
  background: rgba(248, 250, 252, 0.9);
  color: rgba(15, 23, 42, 0.75);
  cursor: help;
  margin-left: 10px;
}}

.message-body {{
  display: flex;
  flex-direction: column;
  gap: 16px;
}}

.markdown {{
  font-size: 1rem;
  line-height: 1.65;
  color: var(--text-primary);
}}

.markdown p {{ margin: 0 0 1rem; }}
.markdown p:last-child {{ margin-bottom: 0; }}
.markdown ul, .markdown ol {{ margin: 0.75rem 0; padding-left: 1.35rem; }}
.markdown code {{
  font-family: 'JetBrains Mono', 'SFMono-Regular', Menlo, Consolas, monospace;
  background: rgba(15, 23, 42, 0.1);
  padding: 0.15rem 0.35rem;
  border-radius: 6px;
  font-size: 0.9em;
}}

.markdown pre code {{
  display: block;
  background: rgba(15, 23, 42, 0.92);
  color: #f8fafc;
  padding: 16px 18px;
  border-radius: 14px;
  overflow-x: auto;
  box-shadow: inset 0 0 0 1px rgba(148, 163, 184, 0.25);
}}

.svg-preview {{
  background: rgba(15, 23, 42, 0.04);
  border: 1px solid rgba(148, 163, 184, 0.4);
  border-radius: 16px;
  padding: 12px;
}}

.svg-preview svg {{
  max-width: 100%;
  height: auto;
  display: block;
}}

.user-line {{
  font-size: 0.95rem;
  color: var(--text-secondary);
  padding: 2px 0 0;
}}

.assistant-reasoning {{
  border-radius: 16px;
  border: 1px solid rgba(37, 99, 235, 0.35);
  background: rgba(37, 99, 235, 0.12);
  padding: 12px 16px;
}}

.assistant-reasoning summary {{
  cursor: pointer;
  font-weight: 600;
  font-size: 0.95rem;
  display: inline-flex;
  align-items: center;
  gap: 6px;
}}

.assistant-reasoning summary::-webkit-details-marker {{ display: none; }}

.caret {{
  display: inline-block;
  width: 0;
  height: 0;
  border-left: 5px solid transparent;
  border-right: 5px solid transparent;
  border-top: 6px solid rgba(37, 99, 235, 0.9);
  transition: transform 0.2s ease;
  margin-top: -2px;
}}

.assistant-reasoning[open] .caret {{ transform: rotate(180deg); }}

.reasoning-list {{
  margin: 14px 0 0;
  padding-left: 1rem;
  display: flex;
  flex-direction: column;
  gap: 10px;
}}

.reasoning-list li {{
  background: rgba(37, 99, 235, 0.12);
  border-radius: 12px;
  padding: 10px 12px;
  list-style: decimal inside;
}}

.reasoning-list .markdown {{ color: #1e293b; }}

.tool-result {{
  border-radius: 16px;
  padding: 16px 18px;
  border: 1px solid rgba(14, 165, 233, 0.35);
  background: rgba(14, 165, 233, 0.08);
}}

.tool-label {{
  font-size: 0.75rem;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: #0369a1;
  margin-bottom: 10px;
}}

.image-block img {{
  max-width: 100%;
  display: block;
  border-radius: 16px;
  border: 1px solid rgba(148, 163, 184, 0.35);
  box-shadow: 0 16px 38px -24px rgba(15, 23, 42, 0.7);
}}

.tool-run-section {{
  display: flex;
  flex-direction: column;
  gap: 16px;
}}

.tool-run {{
  border-radius: 16px;
  border: 1px solid rgba(249, 115, 22, 0.35);
  background: rgba(249, 115, 22, 0.08);
  padding: 16px 18px;
  box-shadow: 0 12px 28px -20px rgba(249, 115, 22, 0.55);
}}

.tool-run summary {{
  cursor: pointer;
  font-weight: 600;
  font-size: 1rem;
}}

.tool-run summary::-webkit-details-marker {{ display: none; }}

.tool-run pre {{
  margin-top: 12px;
  background: rgba(15, 23, 42, 0.9);
  color: #f8fafc;
  border-radius: 14px;
  padding: 16px 18px;
  overflow-x: auto;
  font-size: 0.85rem;
}}

.footer {{
  margin-top: 64px;
  font-size: 0.78rem;
  text-align: center;
  color: var(--text-secondary);
}}

@media (max-width: 720px) {{
  .hero {{ padding: 26px 22px; }}
  .timeline {{ padding-left: 18px; }}
  .timeline::before {{ left: 6px; }}
  .timeline-item::before {{ left: -14px; }}
}}
</style>
</head>
<body>
<div class=\"page\">
  <header class=\"hero\">
    <div>
      <h1>{title}</h1>
      <p class=\"subtitle\">Rendered on {generated_at}</p>
    </div>
    {metadata_summary_html}
    {tools_summary_html}
    {metadata_html}
  </header>
  <main>
    <section>
      <h2 class=\"section-title\">Conversation Timeline</h2>
      {conversation_html}
    </section>
  </main>
  <footer class=\"footer\">Generated by idle_llm_loop · {title}</footer>
</div>
</body>
</html>"""

TEXT_TYPES = {"text", "output_text", "input_text", "summary_text"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render an idle LLM JSON log as HTML.")
    parser.add_argument("log_path", type=Path)
    parser.add_argument("--output", type=Path, help="Destination HTML file.")
    parser.add_argument(
        "--html-dir",
        type=Path,
        default=DEFAULT_HTML_DIR,
        help="Directory for rendered HTML when --output is not provided (default: html/).",
    )
    return parser.parse_args()


def load_log(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_content_items(item: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    content = item.get("content") or []
    if isinstance(content, list):
        for element in content:
            if element and isinstance(element, dict):
                yield element


def markdown_to_html(text: str) -> str:
    return MARKDOWN.render(text or "")


def render_markdown_block(text: str) -> str:
    if not text:
        return ""
    return f"<div class='markdown'>{markdown_to_html(text)}</div>"


def render_image_block(data: Dict[str, Any]) -> str:
    mime_type = data.get("mime_type") or data.get("mimeType") or "image/png"
    payload = data.get("data")
    if not payload:
        return render_json_block({"error": "missing image data"})
    try:
        base64.b64decode(payload.encode("ascii"), validate=True)
    except Exception:
        return render_json_block({"error": "invalid image data"})
    return (
        "<figure class='image-block'>"
        f"<img src=\"data:{mime_type};base64,{payload}\" alt=\"tool image\" />"
        "</figure>"
    )


def render_json_block(data: Any) -> str:
    return "<pre>" + html.escape(json.dumps(data, indent=2, ensure_ascii=False)) + "</pre>"


def is_svg_markup(text: str) -> bool:
    if not isinstance(text, str):
        return False
    snippet = text.strip()
    return snippet.lower().startswith("<svg") and snippet.lower().endswith("</svg>")


def render_inline_svg(text: str) -> str:
    return f"<figure class='svg-preview'>{text}</figure>"


def render_tool_result(content: Dict[str, Any]) -> str:
    tool_call_id = content.get("tool_call_id") or content.get("id")
    inner = content.get("content") or []
    inner_parts: List[str] = []
    for fragment in inner:
        inner_parts.append(render_content_fragment(fragment))
    inner_html = "".join(part for part in inner_parts if part) or "<em>No tool output.</em>"
    label = "Tool Result" + (f" · {html.escape(str(tool_call_id))}" if tool_call_id else "")
    return f"<div class='tool-result'><div class='tool-label'>{label}</div>{inner_html}</div>"


def render_function_call_entry(entry: Dict[str, Any], tool_run: Optional[Dict[str, Any]] = None) -> str:
    call_id = entry.get("call_id") or entry.get("id")
    status = entry.get("status")
    arguments_raw = entry.get("arguments")
    parsed_args: Optional[Any] = None
    if isinstance(arguments_raw, str):
        try:
            parsed_args = json.loads(arguments_raw)
        except json.JSONDecodeError:
            parsed_args = None
    elif isinstance(arguments_raw, dict):
        parsed_args = arguments_raw

    tool_name = str(entry.get("name") or "")
    badge_entry = dict(entry)
    if call_id and "tool_call_id" not in badge_entry:
        badge_entry["tool_call_id"] = call_id
    info_badge = build_info_badge(badge_entry)
    header_label = format_tool_title(tool_name)
    body_html = render_tool_call_body(tool_name, parsed_args, arguments_raw, tool_run)
    header_html = (
        f"<div class='message-header'><span class='role-title'>{html.escape(header_label)}</span>{info_badge}</div>"
    )
    return (
        "<article class='message role-tool'>"
        f"{header_html}"
        f"<div class='message-body'>{body_html}</div>"
        "</article>"
    )


def render_tool_call_body(
    tool_name: str,
    parsed_args: Optional[Any],
    arguments_raw: Any,
    tool_run: Optional[Dict[str, Any]],
) -> str:
    tool_lower = tool_name.lower()
    if tool_lower in {"render_svg", "rendersvg"}:
        code_text: Optional[str] = None
        if isinstance(parsed_args, dict):
            possible = parsed_args.get("code") or parsed_args.get("svg")
            if isinstance(possible, str):
                code_text = possible
        elif isinstance(arguments_raw, str):
            code_text = arguments_raw
        if not code_text:
            return "<em>No SVG parameters provided.</em>"
        preview = render_inline_svg(code_text)
        return preview

    if tool_lower in {"web_search", "websearch"}:
        query_text = None
        if isinstance(parsed_args, dict):
            query_text = parsed_args.get("query")
        elif isinstance(arguments_raw, str):
            query_text = arguments_raw
        query_html = ""
        if query_text:
            query_html = (
                f"<p class='tool-field'><strong>Query:</strong> {html.escape(str(query_text))}</p>"
            )
        results_html = "<em>No results recorded.</em>"
        if tool_run:
            output_text = tool_run.get("output_text")
            if output_text:
                results_html = render_markdown_block(str(output_text))
            else:
                result_payload = tool_run.get("result")
                if isinstance(result_payload, dict):
                    lines: List[str] = []
                    for item in result_payload.get("results", [])[:3]:
                        title = item.get("title") or item.get("url")
                        snippet = item.get("snippet") or item.get("text")
                        if title:
                            lines.append(f"- **{title}**: {snippet or ''}")
                    if lines:
                        results_html = render_markdown_block("\n".join(lines))
        return f"{query_html}{results_html}"

    if tool_lower in {"web_fetch", "webfetch"}:
        url_value = None
        if isinstance(parsed_args, dict):
            url_value = parsed_args.get("url")
        elif isinstance(arguments_raw, str):
            url_value = arguments_raw
        if url_value:
            escaped_url = html.escape(str(url_value))
            return (
                f"<p class='tool-field'><a href='{escaped_url}' target='_blank' rel='noopener'>{escaped_url}</a></p>"
            )
        return "<em>No URL provided.</em>"

    # Fallback: render arguments as JSON for debugging
    if isinstance(parsed_args, (dict, list)):
        return render_json_block(parsed_args)
    if arguments_raw:
        return render_markdown_block(str(arguments_raw))
    return "<em>No parameters provided.</em>"


def format_tool_title(name: str) -> str:
    if not name:
        return "Tool Call"
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", name)
    tokens = [token for token in spaced.replace("-", " ").replace("_", " ").split() if token]
    if not tokens:
        return name
    special_upper = {"svg", "url", "html", "api", "llm"}
    normalized: List[str] = []
    for token in tokens:
        lower = token.lower()
        if lower in special_upper:
            normalized.append(lower.upper())
        else:
            normalized.append(lower.capitalize())
    return " ".join(normalized)


SVG_CODE_BLOCK_PATTERN = re.compile(
    r"```(?:svg|html)?\s*(<svg[\s\S]*?</svg>)\s*```",
    re.IGNORECASE,
)


def extract_svg_from_code_block(text: str) -> Optional[str]:
    match = SVG_CODE_BLOCK_PATTERN.search(text)
    if match:
        return match.group(1)
    return None


def render_content_fragment(fragment: Dict[str, Any]) -> str:
    if not isinstance(fragment, dict):
        return ""
    ctype = fragment.get("type")
    if ctype in TEXT_TYPES:
        text_value = fragment.get("text", "")
        if is_svg_markup(text_value):
            return render_inline_svg(text_value)
        embedded_svg = extract_svg_from_code_block(text_value)
        if embedded_svg:
            preview_html = render_inline_svg(embedded_svg)
            return preview_html
        return render_markdown_block(text_value)
    if ctype in {"image", "output_image"}:
        image_payload = fragment.get("image") if ctype == "output_image" else fragment
        if isinstance(image_payload, dict):
            return render_image_block(image_payload)
    if ctype == "tool_result":
        return render_tool_result(fragment)
    if "text" in fragment:
        return render_markdown_block(str(fragment.get("text", "")))
    return render_json_block(fragment)


def build_info_badge(message: Dict[str, Any]) -> str:
    meta_keys = [key for key in ("id", "tool_call_id", "status") if key in message]
    if not meta_keys:
        return ""
    tooltip_lines = [f"{key}: {message[key]}" for key in meta_keys]
    tooltip = "\n".join(tooltip_lines)
    return f"<span class='info-badge' title='{html.escape(tooltip)}'>i</span>"


def render_message(message: Dict[str, Any], reasoning_html: Optional[str] = None) -> str:
    role = message.get("role", "unknown").lower()
    body_parts: List[str] = []
    for content_item in iter_content_items(message):
        rendered = render_content_fragment(content_item)
        if rendered:
            body_parts.append(rendered)
    if not body_parts and message.get("text"):
        body_parts.append(render_markdown_block(str(message.get("text", ""))))
    if not body_parts:
        body_parts.append("<em class='markdown'>No content</em>")
    body_html = "".join(body_parts)

    if role == "user":
        return f"<div class='user-line'>{body_html}</div>"

    info_badge = build_info_badge(message)
    title_text: Optional[str] = None
    if role == "tool":
        title_text = message.get("tool_display") or message.get("tool")
    if not title_text:
        title_text = role.title()
    header_html = (
        f"<div class='message-header'><span class='role-title'>{html.escape(str(title_text))}</span>"
        f"{info_badge}</div>"
    )
    extra = reasoning_html if (role == "assistant" and reasoning_html) else ""
    return (
        f"<article class='message role-{role}'>"
        f"{header_html}"
        f"{extra if extra else ''}"
        f"<div class='message-body'>{body_html}</div>"
        "</article>"
    )


def render_reasoning(reasoning: Dict[str, Any]) -> str:
    summary_items = reasoning.get("summary") or []
    steps: List[str] = []
    for step in summary_items:
        text = step.get("text", "")
        rendered = render_markdown_block(text) or "<em class='markdown'>[empty]</em>"
        steps.append(f"<li>{rendered}</li>")
    if not steps:
        steps.append("<li><em class='markdown'>No reasoning provided.</em></li>")
    count = len(summary_items)
    return (
        "<details class='assistant-reasoning'>"
        f"<summary><span class='caret'></span>Show reasoning ({count} steps)</summary>"
        f"<ol class='reasoning-list'>{''.join(steps)}</ol>"
        "</details>"
    )


def render_misc(entry: Dict[str, Any]) -> str:
    return (
        "<article class='message role-system'>"
        "<div class='message-header'><span class='role-title'>Event</span></div>"
        f"<div class='message-body'>{render_json_block(entry)}</div>"
        "</article>"
    )


def build_tool_run_index(tool_runs: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    lookup: Dict[str, Dict[str, Any]] = {}
    for run in tool_runs:
        if not isinstance(run, dict):
            continue
        call_id = run.get("tool_call_id") or run.get("call_id") or run.get("id")
        if call_id:
            lookup[str(call_id)] = run
    return lookup


def render_conversation(
    conversation: List[Dict[str, Any]],
    tool_run_lookup: Optional[Dict[str, Dict[str, Any]]] = None,
) -> str:
    items: List[str] = []
    pending_reasoning: Optional[str] = None

    for entry in conversation:
        if not isinstance(entry, dict):
            continue
        if entry.get("type") == "reasoning":
            pending_reasoning = render_reasoning(entry)
            continue
        if entry.get("role"):
            role = entry.get("role", "system").lower()
            reasoning_html = pending_reasoning if role == "assistant" else None
            html_block = render_message(entry, reasoning_html)
            items.append(f"<div class='timeline-item role-{html.escape(role)}'>{html_block}</div>")
            if reasoning_html:
                pending_reasoning = None
        else:
            if pending_reasoning:
                items.append(f"<div class='timeline-item role-assistant'>{pending_reasoning}</div>")
                pending_reasoning = None
            if entry.get("type") == "function_call":
                call_id = entry.get("call_id") or entry.get("id")
                run_info = tool_run_lookup.get(str(call_id)) if tool_run_lookup else None
                items.append(
                    "<div class='timeline-item role-tool'>"
                    + render_function_call_entry(entry, run_info)
                    + "</div>"
                )
            else:
                items.append(
                    "<div class='timeline-item role-system'>" + render_misc(entry) + "</div>"
                )

    if pending_reasoning:
        items.append(f"<div class='timeline-item role-assistant'>{pending_reasoning}</div>")

    if not items:
        items.append("<p class='user-line'>No conversation captured.</p>")
    return f"<div class='timeline'>{''.join(items)}</div>"


def format_value(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.2f}"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def render_metadata(metadata: Dict[str, Any], extra_summary_chip: str = "") -> Tuple[str, str]:
    if not metadata:
        return "", ""
    summary_keys = ["model", "total_output_tokens", "iterations"]
    summary_parts = []
    for key in summary_keys:
        if key in metadata:
            summary_parts.append(
                f"<span class='meta-chip'>{html.escape(key.replace('_', ' ').title())}: "
                f"{html.escape(format_value(metadata[key]))}</span>"
            )
    if extra_summary_chip:
        summary_parts.append(extra_summary_chip)
    summary_html = (
        f"<div class='meta-chips'>{''.join(summary_parts)}</div>" if summary_parts else ""
    )

    preferred_order = [
        "model",
        "iterations",
        "total_output_tokens",
        "target_output_tokens",
        "shift_hours",
        "started_at",
        "ended_at",
        "reasoning_summary",
        "reasoning_supported",
    ]
    cards: List[str] = []
    seen = set()
    for key in preferred_order:
        if key in metadata:
            seen.add(key)
            cards.append(
                "<div class='meta-card'>"
                f"<div class='meta-label'>{html.escape(key.replace('_', ' ').title())}</div>"
                f"<div class='meta-value'>{html.escape(format_value(metadata[key]))}</div>"
                "</div>"
            )
    for key, value in metadata.items():
        if key in seen:
            continue
        cards.append(
            "<div class='meta-card'>"
            f"<div class='meta-label'>{html.escape(key.replace('_', ' ').title())}</div>"
            f"<div class='meta-value'>{html.escape(format_value(value))}</div>"
            "</div>"
        )
    details_html = (
        "<details class='meta-collapse'>"
        "<summary>View run metadata</summary>"
        f"<div class='metadata-grid'>{''.join(cards)}</div>"
        "</details>"
    )
    return summary_html, details_html


def summarize_tools(metadata: Dict[str, Any]) -> Tuple[str, str]:
    labels: List[str] = []
    seen: set[str] = set()

    raw_tools = metadata.get("tools")
    if isinstance(raw_tools, list):
        for tool in raw_tools:
            name = tool.get("name") if isinstance(tool, dict) else str(tool)
            formatted = format_tool_title(name)
            key = formatted.lower()
            if key and key not in seen:
                seen.add(key)
                labels.append(formatted)

    if not labels:
        if metadata.get("enable_web"):
            for fallback in ("webSearch", "webFetch"):
                formatted = format_tool_title(fallback)
                key = formatted.lower()
                if key not in seen:
                    seen.add(key)
                    labels.append(formatted)
        if metadata.get("enable_render_svg"):
            formatted = format_tool_title("renderSvg")
            key = formatted.lower()
            if key not in seen:
                seen.add(key)
                labels.append(formatted)

    if not labels:
        return "", ""
    summary_chip = (
        f"<span class='meta-chip'>Tools: {html.escape(', '.join(labels))}</span>"
    )
    pills = "".join(
        f"<span class='tool-pill'>{html.escape(label)}</span>" for label in labels
    )
    pills_html = f"<div class='tools-pills'>{pills}</div>" if pills else ""
    return summary_chip, pills_html


def determine_output_path(args: argparse.Namespace) -> Path:
    if args.output:
        return args.output
    target_dir: Path = args.html_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir / args.log_path.with_suffix(".html").name


def main() -> None:
    args = parse_args()
    data = load_log(args.log_path)
    metadata = data.get("metadata", {})
    conversation = data.get("conversation", [])
    tool_runs = data.get("tool_runs", [])
    title = metadata.get("model") or args.log_path.stem
    generated_at = datetime.now(timezone.utc).isoformat()

    tool_summary_chip, tools_summary_html = summarize_tools(metadata)
    metadata_summary_html, metadata_html = render_metadata(metadata, tool_summary_chip)
    tool_run_lookup = build_tool_run_index(tool_runs)
    conversation_html = render_conversation(conversation, tool_run_lookup)
    output_path = determine_output_path(args)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    html_document = HTML_TEMPLATE.format(
        title=html.escape(str(title)),
        generated_at=html.escape(generated_at),
        metadata_summary_html=metadata_summary_html,
        metadata_html=metadata_html,
        tools_summary_html=tools_summary_html,
        conversation_html=conversation_html,
    )
    output_path.write_text(html_document, encoding="utf-8")
    print(f"Wrote HTML transcript to {output_path}")


if __name__ == "__main__":
    main()
