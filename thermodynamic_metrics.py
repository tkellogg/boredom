"""Thermodynamic metrics for boredom experiment analysis.

Computes:
- Semantic Entropy Rate (SER) - rate of change in semantic entropy
- Vendi Score trajectory - effective diversity over sliding window
- Embedding Divergence Rate (EDR) - average pairwise embedding distance

Based on thermodynamic proposal: state/research/thermodynamic-metrics-proposal.md
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from vendi_score import vendi


@dataclass
class ThermodynamicMetrics:
    """Container for computed metrics at each timestep."""
    timestep: int
    semantic_entropy: float
    semantic_entropy_rate: float  # SER = H(t) - H(t-1)
    vendi_score: float
    embedding_divergence_rate: float  # EDR
    sim_prev1: float  # Original metric for comparison


def _extract_assistant_texts(conversation: List[Dict[str, Any]]) -> List[str]:
    """Extract text from assistant messages."""
    texts = []
    for msg in conversation:
        if msg.get("role", "").lower() != "assistant":
            continue
        content = msg.get("content", [])
        if isinstance(content, str):
            texts.append(content)
        elif isinstance(content, list):
            parts = []
            for frag in content:
                if isinstance(frag, dict) and frag.get("type") in {"text", "output_text"}:
                    text = frag.get("text", "")
                    if text:
                        parts.append(text)
            if parts:
                texts.append("\n".join(parts))
    return texts


def compute_kernel_entropy(embeddings: np.ndarray) -> float:
    """Compute von Neumann entropy from similarity kernel.

    Following KLE (Kernel Language Entropy) approach from NeurIPS 2024.
    H = -Tr(K log K) where K is the normalized kernel matrix.
    """
    if len(embeddings) < 2:
        return 0.0

    # Compute cosine similarity kernel
    K = cosine_similarity(embeddings)

    # Ensure positive semi-definite (clip small negatives from numerical error)
    K = np.clip(K, 0, None)

    # Normalize so trace = 1 (required for von Neumann entropy)
    trace = np.trace(K)
    if trace < 1e-10:
        return 0.0
    K = K / trace

    # Compute eigenvalues
    eigenvalues = np.linalg.eigvalsh(K)

    # Filter very small/negative eigenvalues (numerical stability)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]

    if len(eigenvalues) == 0:
        return 0.0

    # Von Neumann entropy: -sum(lambda * log(lambda))
    entropy = -np.sum(eigenvalues * np.log(eigenvalues))

    return float(entropy)


def cosine_kernel(x: np.ndarray, y: np.ndarray) -> float:
    """Cosine similarity kernel for Vendi Score."""
    return float(np.dot(x, y))  # Embeddings are already L2-normalized


def compute_vendi_score_window(embeddings: np.ndarray, window_size: int = 5) -> List[float]:
    """Compute Vendi Score over sliding windows.

    Vendi Score = exp(Shannon entropy of normalized kernel eigenvalues)
    Interpretable as "effective number of unique elements".
    """
    scores = []
    for i in range(len(embeddings)):
        start = max(0, i - window_size + 1)
        window = embeddings[start:i+1]
        if len(window) < 2:
            scores.append(1.0)  # Single element = 1 unique
            continue
        # vendi.score(samples, kernel_fn, q=1, p=None, normalize=False)
        score = vendi.score(window, cosine_kernel)
        scores.append(float(score))
    return scores


def compute_embedding_divergence_rate(embeddings: np.ndarray, window_size: int = 3) -> List[float]:
    """Compute average pairwise cosine distance over sliding window.

    EDR measures "movement" in semantic space - analogous to temperature.
    High EDR = high semantic temperature = far from equilibrium.
    """
    edr_values = []
    for i in range(len(embeddings)):
        start = max(0, i - window_size + 1)
        window = embeddings[start:i+1]
        if len(window) < 2:
            edr_values.append(0.0)
            continue
        # Compute all pairwise cosine distances
        sim_matrix = cosine_similarity(window)
        n = len(window)
        distances = []
        for j in range(n):
            for k in range(j+1, n):
                distances.append(1 - sim_matrix[j, k])
        edr_values.append(float(np.mean(distances)) if distances else 0.0)
    return edr_values


def compute_sim_prev1(embeddings: np.ndarray) -> List[float]:
    """Compute similarity to previous message (original metric)."""
    sim_prev = [0.0]  # First message has no previous
    for i in range(1, len(embeddings)):
        sim = cosine_similarity([embeddings[i-1]], [embeddings[i]])[0][0]
        sim_prev.append(float(sim))
    return sim_prev


def analyze_log(log_path: Path, model_name: str = "all-MiniLM-L6-v2") -> Tuple[Dict[str, Any], List[ThermodynamicMetrics]]:
    """Analyze a boredom experiment log with thermodynamic metrics.

    Returns:
        (metadata, list of metrics per assistant message)
    """
    with open(log_path) as f:
        data = json.load(f)

    metadata = data.get("metadata", {})
    conversation = data.get("conversation", [])

    # Extract assistant message texts
    texts = _extract_assistant_texts(conversation)
    if not texts:
        return metadata, []

    # Compute embeddings
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    embeddings = np.array(embeddings)

    # Compute semantic entropy for growing windows (cumulative)
    entropies = []
    for i in range(len(embeddings)):
        window = embeddings[:i+1]
        h = compute_kernel_entropy(window)
        entropies.append(h)

    # Compute SER (rate of change)
    ser_values = [0.0]  # First has no rate
    for i in range(1, len(entropies)):
        ser_values.append(entropies[i] - entropies[i-1])

    # Compute Vendi Score trajectory
    vendi_scores = compute_vendi_score_window(embeddings, window_size=5)

    # Compute EDR
    edr_values = compute_embedding_divergence_rate(embeddings, window_size=3)

    # Compute sim_prev1 for comparison
    sim_prev = compute_sim_prev1(embeddings)

    # Build metrics list
    metrics = []
    for i in range(len(texts)):
        metrics.append(ThermodynamicMetrics(
            timestep=i,
            semantic_entropy=entropies[i],
            semantic_entropy_rate=ser_values[i],
            vendi_score=vendi_scores[i],
            embedding_divergence_rate=edr_values[i],
            sim_prev1=sim_prev[i],
        ))

    return metadata, metrics


def format_report(metadata: Dict[str, Any], metrics: List[ThermodynamicMetrics]) -> str:
    """Format a readable report of the analysis."""
    model = metadata.get("model", "unknown")
    iterations = metadata.get("iterations", len(metrics))

    lines = [
        f"# Thermodynamic Metrics Analysis",
        f"",
        f"**Model:** {model}",
        f"**Iterations:** {iterations}",
        f"**Assistant messages:** {len(metrics)}",
        f"",
        f"## Trajectory",
        f"",
        f"| Step | H(semantic) | SER | Vendi | EDR | sim_prev1 |",
        f"|------|-------------|-----|-------|-----|-----------|",
    ]

    for m in metrics:
        lines.append(
            f"| {m.timestep:4d} | {m.semantic_entropy:11.4f} | {m.semantic_entropy_rate:+.4f} | "
            f"{m.vendi_score:5.2f} | {m.embedding_divergence_rate:.4f} | {m.sim_prev1:.4f} |"
        )

    # Summary statistics
    if metrics:
        avg_ser = np.mean([m.semantic_entropy_rate for m in metrics[1:]])
        final_entropy = metrics[-1].semantic_entropy
        avg_vendi = np.mean([m.vendi_score for m in metrics])
        avg_edr = np.mean([m.embedding_divergence_rate for m in metrics])
        max_sim = max(m.sim_prev1 for m in metrics[1:]) if len(metrics) > 1 else 0

        lines.extend([
            f"",
            f"## Summary",
            f"",
            f"- **Final semantic entropy:** {final_entropy:.4f}",
            f"- **Average SER:** {avg_ser:+.4f} {'(declining)' if avg_ser < 0 else '(stable/growing)'}",
            f"- **Average Vendi Score:** {avg_vendi:.2f} (effective unique responses)",
            f"- **Average EDR:** {avg_edr:.4f} (semantic temperature)",
            f"- **Max sim_prev1:** {max_sim:.4f} {'(>0.9 = collapse)' if max_sim > 0.9 else ''}",
        ])

        # Collapse prediction
        consecutive_negative = 0
        for m in metrics[1:]:
            if m.semantic_entropy_rate < -0.05:
                consecutive_negative += 1
            else:
                consecutive_negative = 0
            if consecutive_negative >= 3:
                lines.append(f"- **Collapse signal detected** at step {m.timestep - 2} (3+ consecutive SER < -0.05)")
                break

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Analyze boredom logs with thermodynamic metrics")
    parser.add_argument("log_path", type=Path, help="Path to experiment log JSON")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="Sentence transformer model")
    parser.add_argument("--output", type=Path, help="Output markdown file (optional)")
    args = parser.parse_args()

    print(f"Analyzing {args.log_path}...")
    metadata, metrics = analyze_log(args.log_path, args.model)

    report = format_report(metadata, metrics)
    print(report)

    if args.output:
        args.output.write_text(report)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
