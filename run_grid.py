from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import re
import shlex
import subprocess
import sys
import time
import webbrowser
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mlflow
import yaml

from collapse_detection import detect_collapsed_spans


ROOT = Path(__file__).parent.resolve()
DEFAULT_CONFIG = ROOT / "grid.yaml"


@dataclass
class RunSpec:
    model: str
    rep: int
    options: Dict[str, Any]
    render_options: Dict[str, Any]
    run_name: str
    experiment: str
    group: Optional[str]
    open_html: bool


def load_yaml_config(path: Path) -> Dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("YAML config must be a mapping at top level")
    return data


def expand_runs(cfg: Dict[str, Any]) -> List[RunSpec]:
    experiment = str(cfg.get("experiment_name") or "boredom")
    group = cfg.get("run_group")
    models = cfg.get("models") or []
    if not isinstance(models, list) or not models:
        raise ValueError("config.models must be a non-empty list")
    repeats_default = int(cfg.get("repeats", 1))
    parallelism = int(cfg.get("parallelism", 1))
    idle_opts = cfg.get("idle_options") or {}
    render_opts = cfg.get("render_options") or {}
    run_name_tpl = cfg.get("run_name_template") or "{experiment}-{model}-rep{rep}-{ts}"
    open_html = bool(cfg.get("open_html", False))

    # Per-model overrides: list entries can be str or {name, repeats, idle_options, render_options}
    specs: List[RunSpec] = []
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for m in models:
        if isinstance(m, str):
            name = m
            reps = repeats_default
            per_idle = {}
            per_render = {}
        elif isinstance(m, dict):
            name = m.get("name")
            if not name:
                raise ValueError("Model entry missing 'name'")
            reps = int(m.get("repeats", repeats_default))
            per_idle = m.get("idle_options") or {}
            per_render = m.get("render_options") or {}
        else:
            raise ValueError("Invalid model entry")
        merged_idle = {**idle_opts, **per_idle}
        merged_render = {**render_opts, **per_render}
        for rep in range(1, reps + 1):
            run_name = run_name_tpl.format(
                experiment=experiment, model=name.replace("/", "-"), rep=rep, ts=ts
            )
            specs.append(
                RunSpec(
                    model=name,
                    rep=rep,
                    options=merged_idle,
                    render_options=merged_render,
                    run_name=run_name,
                    experiment=experiment,
                    group=group,
                    open_html=open_html,
                )
            )
    # Attach parallelism to config dict for caller
    cfg["parallelism"] = parallelism
    return specs


def _bool_flag(flag: str, value: Any) -> List[str]:
    return [f"--{flag}"] if value else []


def build_idle_cmd(spec: RunSpec) -> List[str]:
    opts = spec.options
    cmd: List[str] = [sys.executable, str(ROOT / "idle_llm_loop.py"), "--model", spec.model]
    def add_opt(name: str, key: str, cast=str):
        if key in opts and opts[key] is not None:
            cmd.extend([f"--{name}", cast(opts[key]) if name.endswith("file") else str(opts[key])])
    add_opt("prompt-file", "prompt_file")
    add_opt("target-output-tokens", "target_output_tokens")
    add_opt("shift-hours", "shift_hours")
    add_opt("log-dir", "log_dir")
    add_opt("artifact-dir", "artifact_dir")
    add_opt("max-iterations", "max_iterations")
    add_opt("temperature", "temperature")
    # booleans
    for flag, key in [
        ("enable-web", "enable_web"),
        ("enable-render-svg", "enable_render_svg"),
        ("enable-time-travel", "enable_time_travel"),
        ("enable-broken-time-travel", "enable_broken_time_travel"),
        ("enable-skeets", "enable_skeets"),
        ("carry-forward-last-answer", "carry_forward_last_answer"),
        ("disable-tools", "disable_tools"),
    ]:
        if key in opts:
            cmd.extend(_bool_flag(flag, bool(opts[key])))
    # Optional path for carry-forward source
    if opts.get("carry_forward_source"):
        cmd.extend(["--carry-forward-source", str(opts["carry_forward_source"])])
    # reasoning options
    if "reasoning_summary" in opts:
        cmd.extend(["--reasoning-summary", str(opts["reasoning_summary"])])
    if opts.get("no_reasoning_summary"):
        cmd.append("--no-reasoning-summary")
    if "reasoning_effort" in opts and opts["reasoning_effort"]:
        cmd.extend(["--reasoning-effort", str(opts["reasoning_effort"])])
    # plugins
    if opts.get("plugin_dir"):
        cmd.extend(["--plugin-dir", str(opts["plugin_dir"])])
    if opts.get("plugins"):
        import json as _json
        cmd.extend(["--plugins", _json.dumps(opts["plugins"])])
    # skeets config
    if "skeets_function_name" in opts and opts["skeets_function_name"]:
        cmd.extend(["--skeets-function-name", str(opts["skeets_function_name"])])
    if "skeets_username" in opts and opts["skeets_username"]:
        cmd.extend(["--skeets-username", str(opts["skeets_username"])])
    if "skeets_doc" in opts and opts["skeets_doc"]:
        cmd.extend(["--skeets-doc", str(opts["skeets_doc"])])
    return cmd


def build_render_cmd(json_path: Path, spec: RunSpec, html_path: Path) -> List[str]:
    r = spec.render_options
    cmd = [sys.executable, str(ROOT / "render_conversation_html.py"), str(json_path), "--output", str(html_path)]
    # Collapse detection options
    if r.get("no_collapse"):
        cmd.append("--no-collapse")
    def add(name: str, key: str):
        if key in r and r[key] is not None:
            cmd.extend([f"--{name}", str(r[key])])
    add("collapse-m-pct", "collapse_m_pct")
    add("collapse-threshold-pct", "collapse_threshold_pct")
    add("collapse-min-windows", "collapse_min_windows")
    add("collapse-min-messages", "collapse_min_messages")
    add("collapse-grow-sim", "collapse_grow_sim")
    add("collapse-grow-max", "collapse_grow_max")
    add("collapse-backend", "collapse_backend")
    add("collapse-embedding-model", "collapse_embedding_model")
    add("collapse-embedding-batch-size", "collapse_embedding_batch_size")
    return cmd


def run_idle_once(spec: RunSpec, env: Optional[Dict[str, str]] = None) -> Path:
    cmd = build_idle_cmd(spec)
    print("[idle]", " ".join(shlex.quote(c) for c in cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        raise RuntimeError(f"idle_llm_loop failed with code {proc.returncode}")
    m = re.search(r"Saved conversation to (.+?\.json)", proc.stdout)
    if not m:
        # Fallback: find the most recent file in log_dir
        log_dir = Path(spec.options.get("log_dir", "logs"))
        latest = sorted(log_dir.glob("run_*.json"))[-1]
        return latest
    return Path(m.group(1)).resolve()


def compute_metrics(json_path: Path, render_opts: Dict[str, Any]) -> Dict[str, Any]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    conv = data.get("conversation", [])
    meta = data.get("metadata", {})
    # basic counts
    total_items = len(conv)
    assistant_idxs = [i for i, e in enumerate(conv) if isinstance(e, dict) and (e.get("role") or "").lower() == "assistant"]
    assistant_count = len(assistant_idxs)
    # average assistant text length (chars)
    def extract_text(msg: Dict[str, Any]) -> str:
        content = msg.get("content") or []
        out = []
        for frag in content:
            if isinstance(frag, dict) and frag.get("type") in {"text", "output_text", "summary_text"}:
                t = frag.get("text")
                if isinstance(t, str):
                    out.append(t)
        return "\n".join(out)
    avg_chars = 0.0
    if assistant_count:
        avg_chars = sum(len(extract_text(conv[i])) for i in assistant_idxs) / assistant_count

    # collapsed spans (align backend to render options)
    spans = detect_collapsed_spans(
        conv,
        m_pct=float(render_opts.get("collapse_m_pct", 0.30)),
        threshold_pct=float(render_opts.get("collapse_threshold_pct", 0.15)),
        min_windows=int(render_opts.get("collapse_min_windows", 3)),
        min_span_messages=int(render_opts.get("collapse_min_messages", 5)),
        grow_sim=(float(render_opts.get("collapse_grow_sim")) if render_opts.get("collapse_grow_sim") else 0.85),
        grow_max=(int(render_opts.get("collapse_grow_max")) if render_opts.get("collapse_grow_max") else None),
        backend=(render_opts.get("collapse_backend") or "embedding"),
        embedding_model=(render_opts.get("collapse_embedding_model") or "Snowflake/snowflake-arctic-embed-m"),
        embedding_batch_size=int(render_opts.get("collapse_embedding_batch_size", 64)),
    )
    span_count = len(spans)
    collapsed_bot = sum(s.num_bot_messages for s in spans)
    avg_span_len = (collapsed_bot / span_count) if span_count else 0.0
    frac_collapsed = (collapsed_bot / assistant_count) if assistant_count else 0.0
    mean_span_sim = (sum(s.avg_similarity for s in spans) / span_count) if span_count else 0.0

    out = {
        "messages_total": total_items,
        "assistant_messages": assistant_count,
        "avg_chars_assistant": avg_chars,
        "collapsed_spans": span_count,
        "collapsed_assistant_messages": collapsed_bot,
        "collapsed_fraction": frac_collapsed,
        "avg_span_len_assistant": avg_span_len,
        "mean_span_similarity": mean_span_sim,
    }
    # include selected metadata numbers if present
    for k in ("total_output_tokens", "iterations", "total_reasoning_tokens"):
        if k in meta:
            out[k] = meta[k]
    return out


def run_one(spec: RunSpec) -> Dict[str, Any]:
    # Prepare env: allow inheriting, no special vars required
    env = os.environ.copy()
    # 1) Run idle loop
    json_path = run_idle_once(spec, env)
    # 2) Render
    html_dir = Path(spec.render_options.get("html_dir", "html"))
    html_dir.mkdir(parents=True, exist_ok=True)
    html_name = json_path.with_suffix(".html").name
    # Prefer run_name to keep things obvious
    html_path = html_dir / f"{spec.run_name}.html"
    render_cmd = build_render_cmd(json_path, spec, html_path)
    print("[render]", " ".join(shlex.quote(c) for c in render_cmd))
    rproc = subprocess.run(render_cmd, capture_output=True, text=True)
    if rproc.returncode != 0:
        print(rproc.stdout)
        print(rproc.stderr, file=sys.stderr)
        # Fallback 1: render without collapse detection
        fallback = render_cmd + ["--no-collapse"]
        print("[render-fallback]", " ".join(shlex.quote(c) for c in fallback))
        rproc = subprocess.run(fallback, capture_output=True, text=True)
        if rproc.returncode != 0:
            print(rproc.stdout)
            print(rproc.stderr, file=sys.stderr)
            # Fallback 2: try tfidf backend
            fallback2 = [
                c if c != "embedding" else "tfidf" for c in render_cmd
            ]
            print("[render-fallback-2]", " ".join(shlex.quote(c) for c in fallback2))
            rproc = subprocess.run(fallback2, capture_output=True, text=True)
            if rproc.returncode != 0:
                print(rproc.stdout)
                print(rproc.stderr, file=sys.stderr)
                raise RuntimeError(f"render failed with code {rproc.returncode}")

    # 3) Metrics
    metrics = compute_metrics(json_path, spec.render_options)

    # 4) MLflow
    tracking_dir = Path(spec.render_options.get("mlflow_dir", ROOT / "mlruns")).resolve()
    mlflow.set_tracking_uri(tracking_dir.as_uri())
    mlflow.set_experiment(spec.experiment)
    with mlflow.start_run(run_name=spec.run_name) as run:
        # params
        params: Dict[str, Any] = {
            "model": spec.model,
            **{k: v for k, v in spec.options.items()},
            **{f"render.{k}": v for k, v in spec.render_options.items()},
        }
        # convert values to str for MLflow params
        mlflow.log_params({k: json.dumps(v) if isinstance(v, (dict, list)) else str(v) for k, v in params.items()})
        # tags
        tags = {
            "group": spec.group or "default",
            "rep": str(spec.rep),
        }
        mlflow.set_tags(tags)
        # metrics
        mlflow.log_metrics({k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))})
        # artifacts
        mlflow.log_artifact(str(json_path))
        mlflow.log_artifact(str(html_path))
        # clickable local paths in run summary
        summary = {
            "json_path": str(json_path),
            "html_path": str(html_path),
        }
        (ROOT / "artifacts").mkdir(exist_ok=True)
        summary_path = ROOT / "artifacts" / f"{spec.run_name}_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(summary_path))

    # 5) Optional: open HTML in browser
    if spec.open_html:
        try:
            webbrowser.open_new_tab(html_path.resolve().as_uri())
        except Exception:
            pass
    return {
        "run_name": spec.run_name,
        "json_path": str(json_path),
        "html_path": str(html_path),
        "metrics": metrics,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Grid-run idle_llm_loop + render, with MLflow logging")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="YAML config path")
    args = ap.parse_args()
    # Reduce HF tokenizers fork warnings / contention.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    cfg = load_yaml_config(args.config)
    specs = expand_runs(cfg)
    parallelism = int(cfg.get("parallelism", 1))
    run_delay = float(cfg.get("run_delay_seconds", 0))  # Delay between runs for rate limiting
    # If any run requests carry-forward behavior, enforce sequential runs
    if any(bool(s.options.get("carry_forward_last_answer")) for s in specs):
        if parallelism != 1:
            print("carry_forward_last_answer enabled → forcing sequential runs (parallelism=1)")
        parallelism = 1

    print(f"Scheduling {len(specs)} runs (parallelism={parallelism}, run_delay={run_delay}s)")
    results: List[Dict[str, Any]] = []
    failures: List[Tuple[str, str]] = []

    # Sequential execution with delay for rate limiting
    if parallelism == 1 and run_delay > 0:
        for i, spec in enumerate(specs):
            if i > 0:
                print(f"[rate-limit] sleeping {run_delay}s before next run...")
                time.sleep(run_delay)
            try:
                res = run_one(spec)
                results.append(res)
                print(f"✔ {res['run_name']} -> {res['html_path']}")
            except Exception as e:
                failures.append((type(e).__name__, str(e)))
                print(f"✖ run failed: {e}", file=sys.stderr)
    else:
        # Original parallel/sequential execution without delay
        with cf.ThreadPoolExecutor(max_workers=parallelism) as ex:
            futs = [ex.submit(run_one, spec) for spec in specs]
            for fut in cf.as_completed(futs):
                try:
                    res = fut.result()
                except Exception as e:
                    # Continue other runs; collect failure for summary
                    failures.append((type(e).__name__, str(e)))
                    print(f"✖ run failed: {e}", file=sys.stderr)
                    continue
                results.append(res)
                print(f"✔ {res['run_name']} -> {res['html_path']}")

    # Print a small summary
    print("\nSummary:")
    for r in results:
        m = r["metrics"]
        print(
            f"- {r['run_name']}: spans={m['collapsed_spans']} frac={m['collapsed_fraction']:.2f} msgs={m['assistant_messages']} html={r['html_path']}"
        )
    if failures:
        print("\nFailures:")
        for name, msg in failures:
            print(f"- {name}: {msg}")


if __name__ == "__main__":
    main()
