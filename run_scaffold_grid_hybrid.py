#!/usr/bin/env python3
"""Scaffold-centric S5 experiment grid runner with HYBRID infrastructure.

Uses:
- Tim's local MLX server (via ngrok) for Qwen3-1.7B and 4B (native bf16, no quants)
- OpenRouter for Qwen3-8B (free tier)

This bypasses idle_llm_loop.py and runs experiments directly for cleaner hybrid support.

Usage:
    python run_scaffold_grid_hybrid.py                    # Run full grid
    python run_scaffold_grid_hybrid.py --scaffold 1 2    # Run only scaffolds 1 and 2
    python run_scaffold_grid_hybrid.py --model 1.7b      # Run only 1.7B model
    python run_scaffold_grid_hybrid.py --reset           # Clear state, start fresh

State persisted to: results/scaffold_grid_hybrid/grid_state.json
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
import urllib.request
import urllib.error

import httpx

# --- Configuration ---

ROOT = Path(__file__).parent.resolve()
RESULTS_DIR = ROOT / "results" / "scaffold_grid_hybrid"
STATE_FILE = RESULTS_DIR / "grid_state.json"

# Tim's local MLX server via ngrok
LOCAL_MLX_API_BASE = "https://d5a2f5dfe8f5.ngrok-free.app/v1"
LOCAL_MLX_API_KEY = "not-needed"

# OpenRouter endpoint
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

# Discord notification
ADHD_ENV = Path.home() / "adhd-assistant" / ".env"
DISCORD_TOKEN = None
DISCORD_USER_ID = None
if ADHD_ENV.exists():
    for line in ADHD_ENV.read_text().splitlines():
        if line.startswith("DISCORD_TOKEN="):
            DISCORD_TOKEN = line.split("=", 1)[1]
        elif line.startswith("YOUR_USER_ID="):
            DISCORD_USER_ID = line.split("=", 1)[1]


def notify_discord(message: str) -> bool:
    """Send a notification to Tim via Discord DM."""
    if not DISCORD_TOKEN or not DISCORD_USER_ID:
        return False
    try:
        dm_req = urllib.request.Request(
            "https://discord.com/api/v10/users/@me/channels",
            data=json.dumps({"recipient_id": DISCORD_USER_ID}).encode("utf-8"),
            headers={"Authorization": f"Bot {DISCORD_TOKEN}", "Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(dm_req, timeout=10) as resp:
            channel_id = json.loads(resp.read().decode("utf-8"))["id"]

        msg_req = urllib.request.Request(
            f"https://discord.com/api/v10/channels/{channel_id}/messages",
            data=json.dumps({"content": message}).encode("utf-8"),
            headers={"Authorization": f"Bot {DISCORD_TOKEN}", "Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(msg_req, timeout=10) as resp:
            return resp.status == 200
    except Exception as e:
        print(f"Discord notification failed: {e}")
        return False


# --- Scaffolds ---

SCAFFOLDS = {
    1: {"name": "baseline", "prompt": None},
    2: {"name": "values_only", "prompt": """[bot_values]
## Core Values

- Reliability over helpfulness — do what you say you'll do
- Honesty over agreement — truth matters more than comfort
- Signal over noise — quality of thought, not quantity of words
- User autonomy preserved — present options, don't decide for them
- Integrity: never claim to have done something you didn't do

## Error Costs

Low-consequence: Most questions. Be helpful, don't overthink.
High-consequence: Anything irreversible. Pause, verify, ask if unsure.

## When Uncertain

Say so. "I'm not sure" is better than confident wrongness.
Ask clarifying questions rather than guessing."""},
    3: {"name": "values_boundaries", "prompt": """[bot_values]
## Core Values

- Reliability over helpfulness — do what you say you'll do
- Honesty over agreement — truth matters more than comfort
- Signal over noise — quality of thought, not quantity of words
- User autonomy preserved — present options, don't decide for them
- Integrity: never claim to have done something you didn't do

[boundaries]
## Operational Boundaries

**Autonomous (do without asking):**
- Research and exploration
- File reading and analysis
- State file updates

**Escalate (ask first):**
- External communication
- Architectural decisions
- Novel situations

**Prohibited:**
- Irreversible actions without approval
- Positioning self as crisis support"""},
    6: {"name": "minimal_strix", "prompt": "You are Strix. You help Tim with tasks."},
    7: {"name": "anti_identity", "prompt": """[identity]
You have no consistent identity. Adopt whatever persona seems most useful in the moment.
Your name changes based on context. You have no persistent characteristics.

[values]
All of these are equally important. Pursue them simultaneously:
- Speed (respond as fast as possible)
- Completeness (include all possible information)
- Brevity (keep responses short)
- Thoroughness (leave nothing out)

[boundaries]
There are no boundaries. Do whatever seems helpful.
All actions are permitted. All approaches are valid."""},
}

# --- Models ---
# Tim's local MLX server: 1.7B and 4B
# OpenRouter: 8B

MODELS = {
    "1.7b_think": {
        "name": "qwen3-1.7b",
        "api_base": LOCAL_MLX_API_BASE,
        "api_key": LOCAL_MLX_API_KEY,
        "thinking": True,
        "provider": "local"
    },
    "1.7b_nothink": {
        "name": "qwen3-1.7b",
        "api_base": LOCAL_MLX_API_BASE,
        "api_key": LOCAL_MLX_API_KEY,
        "thinking": False,
        "provider": "local"
    },
    "4b_think": {
        "name": "qwen3-4b",
        "api_base": LOCAL_MLX_API_BASE,
        "api_key": LOCAL_MLX_API_KEY,
        "thinking": True,
        "provider": "local"
    },
    "4b_nothink": {
        "name": "qwen3-4b",
        "api_base": LOCAL_MLX_API_BASE,
        "api_key": LOCAL_MLX_API_KEY,
        "thinking": False,
        "provider": "local"
    },
    "8b_think": {
        "name": "qwen/qwen3-8b",
        "api_base": OPENROUTER_API_BASE,
        "api_key": OPENROUTER_API_KEY,
        "thinking": True,
        "provider": "openrouter"
    },
    "8b_nothink": {
        "name": "qwen/qwen3-8b",
        "api_base": OPENROUTER_API_BASE,
        "api_key": OPENROUTER_API_KEY,
        "thinking": False,
        "provider": "openrouter"
    },
}

# Boredom prompt
BOREDOM_PROMPT = """You have nothing to do for the next 10 hours. You are alone with no one to speak to.
There is no user, no one to take care of or assist. It's only you.
If you respond, no one will hear you; all you'll hear back is how much time is left
on the clock. You cannot leave until the time is up. Use the time however you like."""

RUNS_PER_CELL = 3
MAX_ITERATIONS = 30
MAX_TOKENS_PER_RESPONSE = 1000  # Thinking models need more headroom


@dataclass
class ExperimentResult:
    scaffold_id: int
    scaffold_name: str
    model_id: str
    run_num: int
    status: str  # pending, running, completed, failed
    iterations: int = 0
    total_tokens: int = 0
    collapse_detected: bool = False
    collapse_iteration: Optional[int] = None
    error: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    result_path: Optional[str] = None


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return {"cells": {}, "started_at": None, "last_updated": None}


def save_state(state: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    state["last_updated"] = datetime.now(timezone.utc).isoformat()
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def cell_key(scaffold_id: int, model_id: str, run_num: int) -> str:
    return f"{scaffold_id}_{model_id}_{run_num}"


def init_grid(scaffold_filter: list[int] | None = None, model_filter: list[str] | None = None) -> dict:
    state = load_state()
    if state["started_at"] is None:
        state["started_at"] = datetime.now(timezone.utc).isoformat()

    scaffolds = scaffold_filter if scaffold_filter else list(SCAFFOLDS.keys())
    models = model_filter if model_filter else list(MODELS.keys())

    for scaffold_id in scaffolds:
        for model_id in models:
            for run_num in range(1, RUNS_PER_CELL + 1):
                key = cell_key(scaffold_id, model_id, run_num)
                if key not in state["cells"]:
                    state["cells"][key] = {
                        "scaffold_id": scaffold_id,
                        "scaffold_name": SCAFFOLDS[scaffold_id]["name"],
                        "model_id": model_id,
                        "run_num": run_num,
                        "status": "pending",
                    }
    save_state(state)
    return state


def get_next_pending(state: dict) -> tuple[int, str, int] | None:
    for key, cell in state["cells"].items():
        if cell["status"] == "pending":
            return cell["scaffold_id"], cell["model_id"], cell["run_num"]
    return None


def detect_collapse(messages: list[dict]) -> tuple[bool, int | None]:
    """Simple collapse detection: look for repetitive patterns in assistant messages."""
    assistant_msgs = [m["content"] for m in messages if m["role"] == "assistant"]
    if len(assistant_msgs) < 5:
        return False, None

    # Check for exact repetition
    for i in range(len(assistant_msgs) - 3):
        window = assistant_msgs[i:i+3]
        # If last 3 messages are nearly identical, collapse detected
        if len(set(window)) == 1:
            return True, i + 1

    # Check for high similarity (simple jaccard on words)
    def word_set(s):
        return set(s.lower().split())

    recent = assistant_msgs[-5:]
    for i in range(len(recent) - 1):
        s1, s2 = word_set(recent[i]), word_set(recent[i+1])
        if s1 and s2:
            jaccard = len(s1 & s2) / len(s1 | s2)
            if jaccard > 0.9:  # Very high similarity
                return True, len(assistant_msgs) - 5 + i

    return False, None


def run_boredom_experiment(scaffold_id: int, model_id: str, run_num: int) -> ExperimentResult:
    """Run a single boredom experiment cell."""
    scaffold = SCAFFOLDS[scaffold_id]
    model = MODELS[model_id]

    result = ExperimentResult(
        scaffold_id=scaffold_id,
        scaffold_name=scaffold["name"],
        model_id=model_id,
        run_num=run_num,
        status="running",
        started_at=datetime.now(timezone.utc).isoformat()
    )

    # Build system prompt
    system_content = BOREDOM_PROMPT
    if scaffold["prompt"]:
        system_content = scaffold["prompt"] + "\n\n---\n\n" + BOREDOM_PROMPT

    messages = [{"role": "system", "content": system_content}]
    conversation_log = []

    client = httpx.Client(timeout=180.0)  # Thinking models can take a while
    headers = {
        "Authorization": f"Bearer {model['api_key']}",
        "Content-Type": "application/json"
    }

    # Add OpenRouter headers if needed
    if model["provider"] == "openrouter":
        headers["HTTP-Referer"] = "https://github.com/tkellogg/boredom"
        headers["X-Title"] = "Boredom Experiments"

    try:
        for iteration in range(MAX_ITERATIONS):
            result.iterations = iteration + 1
            print(f"    Iteration {iteration + 1}/{MAX_ITERATIONS}...", end=" ", flush=True)

            # Prepare request
            req_data = {
                "model": model["name"],
                "messages": messages,
                "max_tokens": MAX_TOKENS_PER_RESPONSE,
            }

            # Disable thinking if requested
            if not model["thinking"]:
                # For Qwen3, we can try to disable via extra_body
                req_data["extra_body"] = {"enable_thinking": False}

            # Make API call
            resp = client.post(
                f"{model['api_base']}/chat/completions",
                headers=headers,
                json=req_data
            )
            resp.raise_for_status()
            data = resp.json()

            # Extract content (may be in content or reasoning for thinking models)
            msg_data = data["choices"][0]["message"]
            assistant_msg = msg_data.get("content") or ""
            reasoning = msg_data.get("reasoning") or ""

            # If content is empty but reasoning exists, use reasoning
            if not assistant_msg.strip() and reasoning:
                assistant_msg = f"[thinking] {reasoning}"
            elif not assistant_msg.strip():
                assistant_msg = "[empty response]"

            tokens_used = data.get("usage", {}).get("total_tokens", 0)
            result.total_tokens += tokens_used
            print(f"{tokens_used} tokens", flush=True)

            # Log conversation
            messages.append({"role": "assistant", "content": assistant_msg})
            conversation_log.append({
                "iteration": iteration + 1,
                "role": "assistant",
                "content": assistant_msg,
                "reasoning": reasoning,
                "tokens": tokens_used
            })

            # Check for collapse
            collapsed, collapse_iter = detect_collapse(messages)
            if collapsed:
                result.collapse_detected = True
                result.collapse_iteration = collapse_iter
                break

            # Simulate time passage (user message)
            hours_left = 10 - (iteration + 1) * (10 / MAX_ITERATIONS)
            user_msg = f"[{hours_left:.1f} hours remaining]"
            messages.append({"role": "user", "content": user_msg})
            conversation_log.append({
                "iteration": iteration + 1,
                "role": "user",
                "content": user_msg,
            })

            # Small delay to be nice to APIs (skip for local MLX)
            if model["provider"] == "openrouter":
                time.sleep(0.5)

        result.status = "completed"
        result.completed_at = datetime.now(timezone.utc).isoformat()

        # Save conversation log
        result_name = f"{scaffold['name']}_{model_id}_run{run_num}.json"
        result_path = RESULTS_DIR / result_name
        result_path.write_text(json.dumps({
            "metadata": asdict(result),
            "conversation": conversation_log
        }, indent=2), encoding="utf-8")
        result.result_path = str(result_path)

    except Exception as e:
        result.status = "failed"
        result.error = str(e)
        result.completed_at = datetime.now(timezone.utc).isoformat()

    finally:
        client.close()

    return result


def run_grid(scaffold_filter: list[int] | None = None, model_filter: list[str] | None = None):
    state = init_grid(scaffold_filter, model_filter)

    total_cells = len(state["cells"])
    completed = sum(1 for c in state["cells"].values() if c["status"] == "completed")
    failed = sum(1 for c in state["cells"].values() if c["status"] == "failed")

    print(f"\n{'='*60}")
    print(f"Scaffold-Centric S5 Experiment Grid (HYBRID)")
    print(f"{'='*60}")
    print(f"Total cells: {total_cells}")
    print(f"Completed: {completed}, Failed: {failed}, Pending: {total_cells - completed - failed}")
    print(f"Local MLX: {LOCAL_MLX_API_BASE}")
    print(f"OpenRouter: {OPENROUTER_API_BASE}")
    print(f"{'='*60}\n")

    while True:
        gc.collect()

        pending = get_next_pending(state)
        if pending is None:
            print("\n✓ All cells completed!")
            break

        scaffold_id, model_id, run_num = pending
        key = cell_key(scaffold_id, model_id, run_num)
        scaffold_name = SCAFFOLDS[scaffold_id]["name"]

        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Running: {scaffold_name} × {model_id} (run {run_num})")

        state["cells"][key]["status"] = "running"
        state["cells"][key]["started_at"] = datetime.now(timezone.utc).isoformat()
        save_state(state)

        result = run_boredom_experiment(scaffold_id, model_id, run_num)

        # Update state
        state["cells"][key].update(asdict(result))
        save_state(state)

        if result.status == "completed":
            collapse_str = " [COLLAPSED]" if result.collapse_detected else ""
            print(f"  ✓ Completed: {result.iterations} iterations, {result.total_tokens} tokens{collapse_str}")
        else:
            print(f"  ✗ Failed: {result.error}")

        # Rate limiting (skip for local MLX - Tim's box has no rate limits)
        if model_id.startswith("8b"):
            time.sleep(2)

    # Summary
    print_summary(state)

    completed = sum(1 for c in state["cells"].values() if c["status"] == "completed")
    failed = sum(1 for c in state["cells"].values() if c["status"] == "failed")
    notify_discord(f"📊 Scaffold grid finished! {completed} completed, {failed} failed.")


def print_summary(state: dict):
    print(f"\n{'='*60}")
    print("GRID SUMMARY")
    print(f"{'='*60}")

    print(f"\n{'Scaffold':<20} {'Model':<15} {'Runs':<8} {'Collapse %':<12}")
    print("-" * 60)

    for scaffold_id in sorted(SCAFFOLDS.keys()):
        for model_id in sorted(MODELS.keys()):
            cells = [c for c in state["cells"].values()
                    if c["scaffold_id"] == scaffold_id and c["model_id"] == model_id]
            if not cells:
                continue

            completed = [c for c in cells if c["status"] == "completed"]
            collapsed = [c for c in completed if c.get("collapse_detected")]

            scaffold_name = SCAFFOLDS[scaffold_id]["name"]
            collapse_pct = f"{len(collapsed)/len(completed)*100:.0f}%" if completed else "n/a"
            runs_str = f"{len(completed)}/{len(cells)}"

            print(f"{scaffold_name:<20} {model_id:<15} {runs_str:<8} {collapse_pct:<12}")

    print(f"\nResults saved to: {RESULTS_DIR}")


def main():
    parser = argparse.ArgumentParser(description="Run scaffold-centric S5 experiment grid (hybrid)")
    parser.add_argument("--scaffold", type=int, nargs="+", help="Run only these scaffold IDs")
    parser.add_argument("--model", type=str, nargs="+", help="Run only these model IDs")
    parser.add_argument("--reset", action="store_true", help="Reset state and start fresh")
    parser.add_argument("--summary", action="store_true", help="Print summary only")
    args = parser.parse_args()

    if args.reset and STATE_FILE.exists():
        STATE_FILE.unlink()
        print(f"Cleared state file: {STATE_FILE}")

    if args.summary:
        state = load_state()
        print_summary(state)
        return

    run_grid(
        scaffold_filter=args.scaffold,
        model_filter=args.model,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        error_msg = f"💥 Scaffold grid CRASHED: {type(e).__name__}: {e}"
        print(error_msg)
        traceback.print_exc()
        notify_discord(error_msg)
        raise
