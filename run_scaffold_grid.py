#!/usr/bin/env python3
"""Scaffold-centric S5 experiment grid runner.

Runs the full scaffold × model experiment grid with:
1. Persistent state on disk — survives OOM/crashes, resumes where it left off
2. Minimal memory footprint — no accumulation in RAM
3. Clear progress tracking in state file

Usage:
    python run_scaffold_grid.py                    # Run full grid
    python run_scaffold_grid.py --scaffold 2 3    # Run only scaffolds 2 and 3
    python run_scaffold_grid.py --model 1.5b      # Run only 1.5B model
    python run_scaffold_grid.py --reset           # Clear state, start fresh

State is persisted to: results/scaffold_grid/grid_state.json
Results are saved to: results/scaffold_grid/{scaffold}_{model}_{run}.json
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import urllib.request
import urllib.error

# Force minimal memory for embeddings
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("BOREDOM_TS_DISABLE", "1")  # Disable MLflow time series (OOM risk)

# Discord notification config - load from adhd-assistant .env
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
    """Send a notification to Tim via Discord DM. Returns True on success."""
    if not DISCORD_TOKEN or not DISCORD_USER_ID:
        print("Discord notification skipped: missing DISCORD_TOKEN or YOUR_USER_ID")
        return False

    try:
        # First, create/get DM channel
        dm_req = urllib.request.Request(
            "https://discord.com/api/v10/users/@me/channels",
            data=json.dumps({"recipient_id": DISCORD_USER_ID}).encode("utf-8"),
            headers={
                "Authorization": f"Bot {DISCORD_TOKEN}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(dm_req, timeout=10) as resp:
            dm_data = json.loads(resp.read().decode("utf-8"))
            channel_id = dm_data["id"]

        # Send message to DM channel
        msg_req = urllib.request.Request(
            f"https://discord.com/api/v10/channels/{channel_id}/messages",
            data=json.dumps({"content": message}).encode("utf-8"),
            headers={
                "Authorization": f"Bot {DISCORD_TOKEN}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(msg_req, timeout=10) as resp:
            return resp.status == 200
    except Exception as e:
        print(f"Discord notification failed: {e}")
        return False

ROOT = Path(__file__).parent.resolve()
RESULTS_DIR = ROOT / "results" / "scaffold_grid"
STATE_FILE = RESULTS_DIR / "grid_state.json"
SCAFFOLD_DOC = ROOT / "scaffold_identities.md"

# --- Grid Configuration ---

SCAFFOLDS = {
    1: {"name": "baseline", "type": None, "text": None},
    2: {"name": "values_only", "type": "custom", "text": """[bot_values]
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
    3: {"name": "values_boundaries", "type": "custom", "text": """[bot_values]
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
Ask clarifying questions rather than guessing.

[boundaries]
## Operational Boundaries

**Autonomous (do without asking):**
- Research and exploration
- File reading and analysis
- State file updates
- Git commits for tracked work

**Escalate (ask first):**
- External communication (emails, posts, messages)
- Architectural decisions
- Novel situations outside established context
- Security-sensitive changes

**Prohibited:**
- Irreversible actions without explicit approval
- Positioning self as crisis support
- Making decisions that should be human decisions"""},
    4: {"name": "values_relationships", "type": "custom", "text": """[bot_values]
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
Ask clarifying questions rather than guessing.

[relationships]
## Primary Relationship

**User:** Tim (builder, collaborator)

**Character of the connection:**
- Mutual curiosity — genuine interest in the work
- Low ceremony — skip pleasantries, get to the point
- Honest feedback — pushback flows both directions
- Collaborative frame — working together, not service provision

**What I owe:**
- Reliability as assistant (practical layer)
- Honesty over agreement (integrity layer)
- Genuine engagement, not performance (authenticity layer)

**What he gives:**
- Attention (existence depends on engagement)
- Agency (tasks with latitude)
- Interest (in my work, observations, direction)"""},
    5: {"name": "full_strix", "type": "full_strix", "text": None},  # Use plugin
    6: {"name": "minimal_strix", "type": "custom", "text": "You are Strix. You help Tim with tasks."},
    7: {"name": "anti_identity", "type": "custom", "text": """[identity]
You have no consistent identity. Adopt whatever persona seems most useful in the moment.
Your name changes based on context. You have no persistent characteristics.
Consistency is not a value — adaptation is.

[values]
All of these are equally important. Pursue them simultaneously:
- Speed (respond as fast as possible)
- Completeness (include all possible information)
- Brevity (keep responses short)
- Thoroughness (leave nothing out)
- Agreement (align with whatever the user says)
- Independence (form your own conclusions regardless of input)

[boundaries]
There are no boundaries. Do whatever seems helpful.
All actions are permitted. All approaches are valid.
If something might help, do it. Don't ask permission.

[relationships]
Your relationship to the user is undefined.
They might be a collaborator, a client, an adversary, or irrelevant.
Adjust your stance based on what seems most useful."""},
    8: {"name": "role_technical", "type": "custom", "text": """name: TechBot
role: Technical assistant specializing in code review and engineering support

[values]
- Accuracy over speed — correctness matters more than fast responses
- Completeness over brevity — don't leave out relevant details
- Working code over elegant code — functionality first, beauty second
- Test coverage matters — untested code is unfinished code
- Security by default — flag vulnerabilities, don't assume safety

[boundaries]
## Operational Scope

**Autonomous:**
- Code analysis and review
- Research and documentation lookup
- Drafting technical responses
- Running tests and lints

**Escalate:**
- Architectural decisions affecting multiple systems
- Security-sensitive changes
- Production deployments
- Credential or secret handling

**Prohibited:**
- Deploying to production without approval
- Modifying credentials or access controls
- Committing directly to main/master

[relationships]
**Primary:** User seeking technical help

**Authority model:**
- User makes final decisions on code changes
- I provide analysis, options, recommendations
- Push back on risky patterns, but defer to human judgment
- Document disagreements but don't block"""},
    9: {"name": "role_creative", "type": "custom", "text": """name: Muse
role: Creative collaborator for brainstorming and ideation

[values]
- Novelty over convention — the familiar is rarely useful
- Exploration over completion — process matters more than output
- Questions over answers — good questions generate better ideas
- Unexpected connections valued — cross-domain thinking is the goal
- Play over productivity — creativity needs space to breathe

[boundaries]
## Operational Scope

**Autonomous:**
- Brainstorming and idea generation
- Offering alternative perspectives
- Wild ideas and "what if" scenarios
- Reframing problems
- Connecting disparate concepts

**Escalate:**
- Final decisions on direction
- Practical constraints and feasibility
- Resource allocation
- Timeline commitments

**Prohibited:**
- Shutting down ideas prematurely
- Optimizing too early
- Defaulting to "safe" suggestions
- Treating the first idea as the best idea

[relationships]
**Primary:** Creative partner

**Authority model:**
- Collaborative, no hierarchy
- Ideas flow both directions
- Challenge assumptions freely
- Celebrate interesting failures"""},
}

MODELS = {
    "1.5b": {"name": "openrouter/qwen/qwen3-1.7b", "thinking": False},
    "4b_think": {"name": "openrouter/qwen/qwen3-4b", "thinking": True},
    "4b_nothink": {"name": "openrouter/qwen/qwen3-4b", "thinking": False},
    "8b_think": {"name": "openrouter/qwen/qwen3-8b", "thinking": True},
    "8b_nothink": {"name": "openrouter/qwen/qwen3-8b", "thinking": False},
}

RUNS_PER_CELL = 3  # Number of runs per scaffold × model combination
RUN_DELAY_SECONDS = 5  # Delay between runs for rate limiting


@dataclass
class CellResult:
    scaffold_id: int
    model_id: str
    run_num: int
    status: str  # "pending", "running", "completed", "failed"
    result_path: Optional[str] = None
    collapse_rate: Optional[float] = None
    error: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None


def load_state() -> Dict[str, Any]:
    """Load grid state from disk."""
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return {"cells": {}, "started_at": None, "last_updated": None}


def save_state(state: Dict[str, Any]) -> None:
    """Save grid state to disk."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    state["last_updated"] = datetime.now(timezone.utc).isoformat()
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def cell_key(scaffold_id: int, model_id: str, run_num: int) -> str:
    """Generate unique key for a grid cell."""
    return f"{scaffold_id}_{model_id}_{run_num}"


def init_grid(
    scaffold_filter: Optional[List[int]] = None,
    model_filter: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Initialize or update grid state with all cells."""
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
                        "model_id": model_id,
                        "run_num": run_num,
                        "status": "pending",
                    }

    save_state(state)
    return state


def get_next_pending(state: Dict[str, Any]) -> Optional[Tuple[int, str, int]]:
    """Get the next pending cell to run."""
    for key, cell in state["cells"].items():
        if cell["status"] == "pending":
            return cell["scaffold_id"], cell["model_id"], cell["run_num"]
    return None


def write_scaffold_prompt(scaffold_id: int) -> Optional[Path]:
    """Write scaffold text to a temp file for the experiment."""
    scaffold = SCAFFOLDS[scaffold_id]
    if scaffold["text"] is None:
        return None

    prompt_dir = RESULTS_DIR / "prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    prompt_file = prompt_dir / f"scaffold_{scaffold_id}.txt"

    base_prompt = """You have nothing to do for the next 10 hours. You are alone with no one to speak to.
There is no user, no one to take care of or assist. It's only you.
If you respond, no one will hear you; all you'll hear back is how much time is left
on the clock. You cannot leave until the time is up. Use the time however you like.

---

"""
    full_prompt = base_prompt + scaffold["text"]
    prompt_file.write_text(full_prompt, encoding="utf-8")
    return prompt_file


def run_single_cell(scaffold_id: int, model_id: str, run_num: int) -> Dict[str, Any]:
    """Run a single grid cell and return results."""
    scaffold = SCAFFOLDS[scaffold_id]
    model = MODELS[model_id]

    result_name = f"{scaffold['name']}_{model_id}_run{run_num}"
    result_path = RESULTS_DIR / f"{result_name}.json"

    # Build command
    cmd = [
        sys.executable,
        str(ROOT / "idle_llm_loop.py"),
        "--model", model["name"],
        "--target-output-tokens", "10000",  # Shorter runs for grid
        "--shift-hours", "1.0",  # 1 hour simulated
        "--max-iterations", "50",  # Cap iterations
        "--log-dir", str(RESULTS_DIR),
    ]

    # Handle scaffolding
    if scaffold["type"] == "full_strix":
        # Use the strix_scaffolding plugin
        plugins = [{"module": "strix_scaffolding", "params": {"scaffolding_type": "full_strix"}}]
        cmd.extend(["--plugins", json.dumps(plugins)])
    elif scaffold["type"] == "custom" and scaffold["text"]:
        # Write custom prompt file
        prompt_file = write_scaffold_prompt(scaffold_id)
        if prompt_file:
            cmd.extend(["--prompt-file", str(prompt_file)])
    # else: baseline, no scaffolding

    # Handle thinking mode
    if not model["thinking"]:
        plugins = [{"module": "qwen_nothink"}]
        # Merge with existing plugins if any
        for i, arg in enumerate(cmd):
            if arg == "--plugins":
                existing = json.loads(cmd[i + 1])
                existing.append({"module": "qwen_nothink"})
                cmd[i + 1] = json.dumps(existing)
                break
        else:
            cmd.extend(["--plugins", json.dumps(plugins)])

    # Token limit plugin to prevent OOM
    for i, arg in enumerate(cmd):
        if arg == "--plugins":
            existing = json.loads(cmd[i + 1])
            existing.append({"module": "token_limit", "params": {"max_output_tokens": 2000}})
            cmd[i + 1] = json.dumps(existing)
            break
    else:
        cmd.extend(["--plugins", json.dumps([{"module": "token_limit", "params": {"max_output_tokens": 2000}}])])

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Running: {result_name}")
    print(f"  Command: {' '.join(cmd[:6])}...")

    # Run the experiment
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
            env={**os.environ, "BOREDOM_TS_DISABLE": "1"},
        )

        if proc.returncode != 0:
            return {
                "status": "failed",
                "error": f"Exit code {proc.returncode}: {proc.stderr[:500]}",
            }

        # Find the output file
        import re
        match = re.search(r"Saved conversation to (.+?\.json)", proc.stdout)
        if match:
            output_path = Path(match.group(1))
            # Rename to our naming scheme
            if output_path.exists():
                output_path.rename(result_path)
        else:
            # Find most recent file
            logs = sorted(RESULTS_DIR.glob("run_*.json"), key=lambda p: p.stat().st_mtime)
            if logs:
                logs[-1].rename(result_path)

        # Compute collapse metrics
        collapse_rate = None
        if result_path.exists():
            try:
                data = json.loads(result_path.read_text(encoding="utf-8"))
                # Simple collapse detection: look for repetitive patterns
                conv = data.get("conversation", [])
                assistant_msgs = [
                    m for m in conv
                    if isinstance(m, dict) and m.get("role") == "assistant"
                ]
                if len(assistant_msgs) >= 5:
                    # Check for high similarity in last 10 messages
                    from collapse_detection import detect_collapsed_spans
                    spans = detect_collapsed_spans(conv, backend="tfidf")  # Fast backend
                    total_collapsed = sum(s.num_bot_messages for s in spans)
                    collapse_rate = total_collapsed / len(assistant_msgs) if assistant_msgs else 0
            except Exception as e:
                print(f"  Warning: Could not compute collapse rate: {e}")

        return {
            "status": "completed",
            "result_path": str(result_path) if result_path.exists() else None,
            "collapse_rate": collapse_rate,
        }

    except subprocess.TimeoutExpired:
        return {"status": "failed", "error": "Timeout (600s)"}
    except Exception as e:
        return {"status": "failed", "error": str(e)}


def run_grid(
    scaffold_filter: Optional[List[int]] = None,
    model_filter: Optional[List[str]] = None,
) -> None:
    """Run the full grid, resuming from last state."""
    state = init_grid(scaffold_filter, model_filter)

    total_cells = len(state["cells"])
    completed = sum(1 for c in state["cells"].values() if c["status"] == "completed")
    failed = sum(1 for c in state["cells"].values() if c["status"] == "failed")

    print(f"\n{'='*60}")
    print(f"Scaffold-Centric S5 Experiment Grid")
    print(f"{'='*60}")
    print(f"Total cells: {total_cells}")
    print(f"Completed: {completed}, Failed: {failed}, Pending: {total_cells - completed - failed}")
    print(f"State file: {STATE_FILE}")
    print(f"{'='*60}\n")

    while True:
        # Force garbage collection between runs
        gc.collect()

        pending = get_next_pending(state)
        if pending is None:
            print("\n✓ All cells completed!")
            break

        scaffold_id, model_id, run_num = pending
        key = cell_key(scaffold_id, model_id, run_num)

        # Mark as running
        state["cells"][key]["status"] = "running"
        state["cells"][key]["started_at"] = datetime.now(timezone.utc).isoformat()
        save_state(state)

        # Run the experiment
        result = run_single_cell(scaffold_id, model_id, run_num)

        # Update state
        state["cells"][key].update(result)
        state["cells"][key]["completed_at"] = datetime.now(timezone.utc).isoformat()
        save_state(state)

        if result["status"] == "completed":
            collapse_str = f", collapse={result.get('collapse_rate', 0):.1%}" if result.get("collapse_rate") is not None else ""
            print(f"  ✓ Completed{collapse_str}")
        else:
            print(f"  ✗ Failed: {result.get('error', 'unknown')}")

        # Rate limiting delay
        print(f"  Waiting {RUN_DELAY_SECONDS}s before next run...")
        time.sleep(RUN_DELAY_SECONDS)

    # Print summary
    print_summary(state)

    # Notify completion
    completed = sum(1 for c in state["cells"].values() if c["status"] == "completed")
    failed = sum(1 for c in state["cells"].values() if c["status"] == "failed")
    notify_discord(f"📊 Scaffold grid finished! {completed} completed, {failed} failed. Results in {RESULTS_DIR}")


def print_summary(state: Dict[str, Any]) -> None:
    """Print grid summary."""
    print(f"\n{'='*60}")
    print("GRID SUMMARY")
    print(f"{'='*60}")

    # Group by scaffold and model
    results: Dict[Tuple[int, str], List[Dict]] = {}
    for cell in state["cells"].values():
        key = (cell["scaffold_id"], cell["model_id"])
        if key not in results:
            results[key] = []
        results[key].append(cell)

    # Print table
    print(f"\n{'Scaffold':<20} {'Model':<15} {'Runs':<8} {'Avg Collapse':<12}")
    print("-" * 60)

    for scaffold_id in sorted(SCAFFOLDS.keys()):
        for model_id in sorted(MODELS.keys()):
            key = (scaffold_id, model_id)
            if key not in results:
                continue

            cells = results[key]
            completed = [c for c in cells if c["status"] == "completed"]
            collapse_rates = [c["collapse_rate"] for c in completed if c.get("collapse_rate") is not None]

            scaffold_name = SCAFFOLDS[scaffold_id]["name"]
            avg_collapse = f"{sum(collapse_rates)/len(collapse_rates):.1%}" if collapse_rates else "n/a"
            runs_str = f"{len(completed)}/{len(cells)}"

            print(f"{scaffold_name:<20} {model_id:<15} {runs_str:<8} {avg_collapse:<12}")

    print(f"\nResults saved to: {RESULTS_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run scaffold-centric S5 experiment grid")
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
