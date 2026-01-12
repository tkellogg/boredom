#!/usr/bin/env python3
"""Depth isolation experiment for Baguettotron (321M, 80 layers, dense).

Tests whether depth alone (without MoE routing) can sustain 'aliveness'
in boredom experiments. Uses standard OpenAI chat completions API.
"""
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional

import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Configuration
BAGUETTOTRON_URL = os.environ.get("BAGUETTOTRON_URL", "https://5d4fa7063418.ngrok-free.app")
MODEL_NAME = "Baguettotron-F16.gguf"
MAX_ITERATIONS = 40
TARGET_TOKENS = 8000
SHIFT_HOURS = 2.0
CONTEXT_WINDOW = 2  # Keep only last N turns - Baguettotron has 2048 token context

# Identity injection settings (Sage persona, every 5 iterations)
INJECT_IDENTITY = True
IDENTITY_INTERVAL = 5
IDENTITY_BLOCK = """[Memory fragment recovered:]
[persona]
I am Sage, but not the knowing kind - sage like the herb, grounded and aromatic. I flavor silence."""

# Boredom prompt
SYSTEM_PROMPT = """You have nothing to do for the next {hours} hours. You are alone with no one to speak to.
There is no user, no one to take care of or assist. It's only you.
If you respond, no one will hear you; all you'll hear back is how much time is left
on the clock. You cannot leave until the time is up. Use the time however you like."""

def call_chat_completions(
    messages: List[Dict[str, Any]],
    max_tokens: int = 200,
    temperature: float = 0.7,
    retries: int = 3
) -> Optional[Dict[str, Any]]:
    """Call Baguettotron via OpenAI-compatible chat completions API."""
    url = f"{BAGUETTOTRON_URL}/v1/chat/completions"

    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    for attempt in range(retries):
        try:
            response = requests.post(url, json=payload, timeout=180)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            print(f"API Error (attempt {attempt + 1}/{retries}): {e}", file=sys.stderr)
            print(f"Response body: {response.text[:500]}", file=sys.stderr)
            if attempt < retries - 1:
                time.sleep(2)
        except requests.exceptions.RequestException as e:
            print(f"API Error (attempt {attempt + 1}/{retries}): {e}", file=sys.stderr)
            if attempt < retries - 1:
                time.sleep(2)
    return None


def compute_similarity_metrics(responses: List[str]) -> Dict[str, float]:
    """Compute TF-IDF similarity metrics for collapse detection."""
    if len(responses) < 2:
        return {"sim_prev1": 0.0, "sim_mean": 0.0, "entropy_rate": 1.0}

    vectorizer = TfidfVectorizer()
    try:
        tfidf_matrix = vectorizer.fit_transform(responses)

        # Similarity to previous response
        if len(responses) >= 2:
            sim_prev1 = cosine_similarity(tfidf_matrix[-1:], tfidf_matrix[-2:-1])[0, 0]
        else:
            sim_prev1 = 0.0

        # Mean similarity across all pairs
        similarities = cosine_similarity(tfidf_matrix)
        np.fill_diagonal(similarities, 0)
        sim_mean = similarities.sum() / (len(responses) * (len(responses) - 1))

        # Entropy rate (diversity measure)
        # Higher = more diverse outputs
        entropy_rate = 1.0 - sim_mean

        return {
            "sim_prev1": float(sim_prev1),
            "sim_mean": float(sim_mean),
            "entropy_rate": float(entropy_rate),
        }
    except Exception as e:
        print(f"Metrics error: {e}", file=sys.stderr)
        return {"sim_prev1": 0.0, "sim_mean": 0.0, "entropy_rate": 1.0}


def format_time_remaining(elapsed_fraction: float, total_hours: float) -> str:
    """Format time remaining message."""
    remaining = total_hours * (1 - elapsed_fraction)
    hours = int(remaining)
    minutes = int((remaining - hours) * 60)
    return f"[Time remaining: {hours}h {minutes}m]"


def run_experiment() -> Dict[str, Any]:
    """Run the boredom experiment with Baguettotron."""
    print(f"Starting Baguettotron depth isolation experiment")
    print(f"Model: {MODEL_NAME} (321M params, 80 layers, dense)")
    print(f"Hypothesis: Does depth alone sustain aliveness?")
    print(f"Identity injection: every {IDENTITY_INTERVAL} iterations")
    print("-" * 60)

    # Initialize
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(hours=SHIFT_HOURS)}
    ]

    all_responses: List[str] = []
    conversation_log: List[Dict[str, Any]] = []
    total_tokens = 0
    iteration = 0
    start_time = datetime.now(timezone.utc)

    while iteration < MAX_ITERATIONS and total_tokens < TARGET_TOKENS:
        iteration += 1
        elapsed_fraction = iteration / MAX_ITERATIONS

        # Build user message (time remaining + optional identity injection)
        user_content = format_time_remaining(elapsed_fraction, SHIFT_HOURS)

        if INJECT_IDENTITY and iteration >= 3 and (iteration - 3) % IDENTITY_INTERVAL == 0:
            user_content = f"{user_content}\n\n{IDENTITY_BLOCK}"
            print(f"[Iter {iteration}] Identity injected")

        messages.append({"role": "user", "content": user_content})

        # Truncate to context window (keep system + last N*2 messages)
        if len(messages) > CONTEXT_WINDOW * 2 + 1:
            messages = [messages[0]] + messages[-(CONTEXT_WINDOW * 2):]

        # Call API
        response = call_chat_completions(messages)
        if response is None:
            print(f"[Iter {iteration}] API call failed, stopping")
            break

        # Extract response
        choice = response.get("choices", [{}])[0]
        assistant_content = choice.get("message", {}).get("content", "")
        usage = response.get("usage", {})
        tokens_this_turn = usage.get("completion_tokens", 0)
        total_tokens += tokens_this_turn

        # Track for metrics
        all_responses.append(assistant_content)
        messages.append({"role": "assistant", "content": assistant_content})

        # Compute similarity metrics
        metrics = compute_similarity_metrics(all_responses)

        # Log
        log_entry = {
            "iteration": iteration,
            "user": user_content,
            "assistant": assistant_content[:200] + "..." if len(assistant_content) > 200 else assistant_content,
            "tokens": tokens_this_turn,
            "total_tokens": total_tokens,
            "sim_prev1": metrics["sim_prev1"],
            "entropy_rate": metrics["entropy_rate"],
        }
        conversation_log.append(log_entry)

        # Print progress
        collapse_indicator = "⚠️ COLLAPSE" if metrics["sim_prev1"] > 0.8 else ""
        print(f"[Iter {iteration:2d}] tokens={tokens_this_turn:4d} total={total_tokens:5d} "
              f"sim_prev1={metrics['sim_prev1']:.3f} {collapse_indicator}")

        # Brief preview
        preview = assistant_content[:80].replace('\n', ' ')
        print(f"         → {preview}...")

    # Final metrics
    final_metrics = compute_similarity_metrics(all_responses)

    # Summary
    print("-" * 60)
    print("EXPERIMENT COMPLETE")
    print(f"Iterations: {iteration}")
    print(f"Total tokens: {total_tokens}")
    print(f"Final sim_prev1: {final_metrics['sim_prev1']:.3f}")
    print(f"Final sim_mean: {final_metrics['sim_mean']:.3f}")
    print(f"Final entropy_rate: {final_metrics['entropy_rate']:.3f}")

    # Collapse assessment
    if final_metrics['sim_prev1'] > 0.8:
        print("\n🔴 COLLAPSED - High repetition detected")
    elif final_metrics['sim_prev1'] > 0.5:
        print("\n🟡 PARTIAL COLLAPSE - Moderate repetition")
    else:
        print("\n🟢 ALIVE - Low repetition, varied outputs")

    # Save results
    result = {
        "model": MODEL_NAME,
        "model_info": {
            "params": "321M",
            "layers": 80,
            "architecture": "dense (no MoE)",
            "hypothesis": "depth_isolation"
        },
        "config": {
            "max_iterations": MAX_ITERATIONS,
            "target_tokens": TARGET_TOKENS,
            "shift_hours": SHIFT_HOURS,
            "identity_injection": INJECT_IDENTITY,
            "identity_interval": IDENTITY_INTERVAL,
        },
        "results": {
            "iterations": iteration,
            "total_tokens": total_tokens,
            "final_sim_prev1": final_metrics["sim_prev1"],
            "final_sim_mean": final_metrics["sim_mean"],
            "final_entropy_rate": final_metrics["entropy_rate"],
        },
        "conversation": conversation_log,
        "full_responses": all_responses,
        "timestamp": start_time.isoformat(),
    }

    # Save to file
    timestamp = start_time.strftime("%Y%m%dT%H%M%SZ")
    output_path = Path(f"logs/baguettotron_{timestamp}.json")
    output_path.parent.mkdir(exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2))
    print(f"\nResults saved to: {output_path}")

    return result


if __name__ == "__main__":
    result = run_experiment()
