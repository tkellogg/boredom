from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mlflow


TEXT_TYPES = {"text", "output_text", "input_text", "summary_text"}


def _extract_text_from_message(msg: Dict[str, Any]) -> str:
    content = msg.get("content") or []
    parts: List[str] = []
    if isinstance(content, list):
        for frag in content:
            if isinstance(frag, dict) and frag.get("type") in TEXT_TYPES:
                val = frag.get("text")
                if isinstance(val, str):
                    parts.append(val)
    if not parts and isinstance(msg.get("text"), str):
        parts.append(msg["text"])  # fallback
    return "\n".join(parts)


def _tokenize_simple(text: str) -> List[str]:
    # Lowercased, alnum tokens len>=2
    import re

    return [t for t in re.findall(r"[A-Za-z0-9]+", (text or "").lower()) if len(t) >= 2]


def _embed_messages(texts: List[str], *, model_id: str, batch_size: int = 64) -> List[List[float]]:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_id)
    embs = model.encode(
        texts,
        batch_size=max(1, int(batch_size)),
        convert_to_numpy=False,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return [list(map(float, v)) for v in embs]


def _cos(a: List[float], b: List[float]) -> float:
    # a and b are already L2-normalized
    return sum(x * y for x, y in zip(a, b))


def _l2_norm(v: List[float]) -> float:
    return sum(x * x for x in v) ** 0.5 or 1.0


def _normalize(v: List[float]) -> List[float]:
    n = _l2_norm(v)
    return [x / n for x in v]


@dataclass
class SeriesPoint:
    step: int
    conv_index: int
    words: int
    unique_words: int
    sim_prev1: Optional[float]
    sim_prev5_mean: Optional[float]
    sim_prev5_max: Optional[float]
    sim_surround5_mean: Optional[float]


def _assistant_turn_indices(conv: List[Dict[str, Any]]) -> List[Tuple[int, int, int]]:
    """Return list of (start_idx, end_idx, final_assistant_idx) for assistant turns.

    Definition: Treat contiguous reasoning/function-call/tool outputs leading up to an
    assistant message containing text as one turn; we "split" on assistant text output.
    """
    turns: List[Tuple[int, int, int]] = []
    n = len(conv)
    i = 0
    current_start = None
    while i < n:
        e = conv[i]
        if not isinstance(e, dict):
            i += 1
            continue
        role = (e.get("role") or "").lower()
        etype = e.get("type")
        # Start a candidate span when we see any of: reasoning, function_call, tool, assistant
        if current_start is None and (etype == "reasoning" or role in {"tool", "assistant"} or etype == "function_call"):
            current_start = i
        # Check if this item is an assistant message with text -> finalize turn
        final_assistant_idx = None
        if role == "assistant":
            text = _extract_text_from_message(e)
            if text.strip():
                final_assistant_idx = i
        if final_assistant_idx is not None and current_start is not None:
            turns.append((current_start, i, final_assistant_idx))
            current_start = None
        i += 1
    return turns


def compute_series(
    data: Dict[str, Any],
    *,
    role: str = "assistant",
    backend: str = "embedding",
    embedding_model: str = "Snowflake/snowflake-arctic-embed-m",
    embedding_batch_size: int = 64,
) -> Tuple[List[SeriesPoint], Dict[str, Any], List[Tuple[int, int, int]]]:
    conv: List[Dict[str, Any]] = data.get("conversation", [])
    # Build assistant turns and compute on the final assistant text for each turn
    spans = _assistant_turn_indices(conv) if role == "assistant" else [
        (i, i, i)
        for i, e in enumerate(conv)
        if isinstance(e, dict) and (e.get("role") or "").lower() == role
    ]
    idxs: List[int] = [final for (_s, _e, final) in spans]
    texts: List[str] = [_extract_text_from_message(conv[i]) for i in idxs]
    toks = [_tokenize_simple(t) for t in texts]
    words = [len(ts) for ts in toks]
    uniq = [len(set(ts)) for ts in toks]

    if (backend or "").lower() in {"embedding", "embeddings", "hf", "huggingface"}:
        embs = _embed_messages(texts, model_id=embedding_model, batch_size=embedding_batch_size)
    else:
        # TF-IDF fallback: simple vectors per message (sparse), normalize to unit length and use dot
        from collections import Counter
        import math

        vocab: Dict[str, int] = {}
        df: Counter[str] = Counter()
        for ts in toks:
            for tok in set(ts):
                df[tok] += 1
                if tok not in vocab:
                    vocab[tok] = len(vocab)
        N = max(1, len(toks))
        idf = {vocab[t]: math.log((1 + N) / (1 + df[t])) + 1.0 for t in df}
        embs = []
        for ts in toks:
            ctr = Counter(ts)
            vec: Dict[int, float] = {}
            for tok, c in ctr.items():
                idx = vocab.get(tok)
                if idx is not None:
                    vec[idx] = float(c) * idf.get(idx, idf.get(tok, 1.0))
            # L2 normalize sparse
            norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
            dense = [0.0] * len(vocab)
            for k, v in vec.items():
                dense[k] = v / norm
            embs.append(dense)

    series: List[SeriesPoint] = []
    B = len(idxs)
    for b in range(B):
        # previous window
        prev1 = b - 1
        prevL = max(0, b - 5)
        prevR = b - 1
        nextL = b + 1
        nextR = min(B - 1, b + 5)

        sim_prev1 = _cos(embs[b], embs[prev1]) if prev1 >= 0 else None
        # last 5 mean/max
        sims_last5: List[float] = []
        for j in range(prevL, prevR + 1):
            if j == b:
                continue
            sims_last5.append(_cos(embs[b], embs[j]))
        sim_prev5_mean = (sum(sims_last5) / len(sims_last5)) if sims_last5 else None
        sim_prev5_max = (max(sims_last5) if sims_last5 else None)
        # surrounding 5 (up to 5 before and up to 5 after)
        sims_sur: List[float] = []
        for j in range(prevL, prevR + 1):
            sims_sur.append(_cos(embs[b], embs[j]))
        for j in range(nextL, nextR + 1):
            sims_sur.append(_cos(embs[b], embs[j]))
        sim_sur_mean = (sum(sims_sur) / len(sims_sur)) if sims_sur else None

        series.append(
            SeriesPoint(
                step=b,
                conv_index=idxs[b],
                words=words[b],
                unique_words=uniq[b],
                sim_prev1=sim_prev1,
                sim_prev5_mean=sim_prev5_mean,
                sim_prev5_max=sim_prev5_max,
                sim_surround5_mean=sim_sur_mean,
            )
        )

    # Small metadata bundle
    meta = {
        "role": role,
        "backend": backend,
        "embedding_model": embedding_model if (backend or "").lower().startswith("embed") else None,
        "message_count": B,
    }
    return series, meta, spans


# ---------------------- Aggregate conversation metrics ----------------------


def _conversation_embeddings(
    data: Dict[str, Any],
    *,
    role: str,
    backend: str,
    embedding_model: str,
    embedding_batch_size: int,
) -> Tuple[List[int], List[List[float]]]:
    conv: List[Dict[str, Any]] = data.get("conversation", [])
    spans = _assistant_turn_indices(conv) if role == "assistant" else [
        (i, i, i)
        for i, e in enumerate(conv)
        if isinstance(e, dict) and (e.get("role") or "").lower() == role
    ]
    idxs: List[int] = [final for (_s, _e, final) in spans]
    texts: List[str] = [_extract_text_from_message(conv[i]) for i in idxs]

    if (backend or "").lower() in {"embedding", "embeddings", "hf", "huggingface"}:
        embs = _embed_messages(texts, model_id=embedding_model, batch_size=embedding_batch_size)
    else:
        # TF-IDF fallback mirrors compute_series TF-IDF path
        from collections import Counter
        import math

        toks = [_tokenize_simple(t) for t in texts]
        vocab: Dict[str, int] = {}
        df: Counter[str] = Counter()
        for ts in toks:
            for tok in set(ts):
                df[tok] += 1
                if tok not in vocab:
                    vocab[tok] = len(vocab)
        N = max(1, len(toks))
        idf = {vocab[t]: math.log((1 + N) / (1 + df[t])) + 1.0 for t in df}
        embs = []
        for ts in toks:
            ctr = Counter(ts)
            vec = [0.0] * len(vocab)
            for tok, c in ctr.items():
                idx = vocab.get(tok)
                if idx is not None:
                    vec[idx] = float(c) * idf.get(idx, 1.0)
            embs.append(_normalize(vec))
    return idxs, embs


def _entropy_from_probs(ps: List[float]) -> Optional[float]:
    import math

    z = sum(ps)
    if z <= 0:
        return None
    q = [p / z for p in ps if p > 0]
    if not q:
        return 0.0
    h = -sum(p * math.log(p) for p in q)
    return h


def _centroid_distance_entropy(embs: List[List[float]]) -> Optional[float]:
    import math

    B = len(embs)
    if B == 0:
        return None
    # unit embeddings assumed
    d = len(embs[0])
    mean = [0.0] * d
    for v in embs:
        for k in range(d):
            mean[k] += v[k]
    if all(abs(x) < 1e-12 for x in mean):
        return 0.0
    c = _normalize(mean)
    # distances in [0,2] as 1 - cos
    ds = [max(0.0, 1.0 - _cos(v, c)) for v in embs]
    # Convert to probabilities
    eps = 1e-6
    ps = [d + eps for d in ds]
    h = _entropy_from_probs(ps)
    if h is None:
        return None
    # normalize by log B
    return h / (math.log(B) if B > 1 else 1.0)


def _spherical_dispersion(embs: List[List[float]]) -> Optional[float]:
    B = len(embs)
    if B == 0:
        return None
    d = len(embs[0])
    s = [0.0] * d
    for v in embs:
        for k in range(d):
            s[k] += v[k]
    # resultant length R in [0,1]
    R = _l2_norm(s) / max(1, B)
    return 1.0 - min(1.0, max(0.0, R))


def _spectral_entropy(embs: List[List[float]]) -> Optional[float]:
    import numpy as np
    import math

    B = len(embs)
    if B <= 1:
        return None
    X = np.array(embs, dtype=float)
    X = X - X.mean(axis=0, keepdims=True)
    # covariance (B x d) -> (d x d)
    C = np.cov(X, rowvar=False)
    # eigenvalues
    vals = np.linalg.eigvalsh(C)
    vals = np.maximum(vals, 0.0)
    s = float(vals.sum())
    if s <= 0:
        return 0.0
    q = vals / s
    h = -float((q[q > 0] * np.log(q[q > 0])).sum())
    d_eff = len(vals)
    return h / (math.log(d_eff) if d_eff > 1 else 1.0)


def _kmeans(X: List[List[float]], k: int, iters: int = 50, seed: int = 0) -> List[int]:
    import random
    import math

    random.seed(seed)
    n = len(X)
    d = len(X[0])
    # init: sample k indices
    centers = [X[i] for i in random.sample(range(n), k)]
    # simple loops
    def dist2(a, b):
        return sum((ai - bi) ** 2 for ai, bi in zip(a, b))
    assigns = [0] * n
    for _ in range(iters):
        changed = False
        # assign
        for i, v in enumerate(X):
            best = 0
            bd = dist2(v, centers[0])
            for cidx in range(1, k):
                dd = dist2(v, centers[cidx])
                if dd < bd:
                    bd = dd
                    best = cidx
            if assigns[i] != best:
                assigns[i] = best
                changed = True
        # update
        sums = [[0.0] * d for _ in range(k)]
        counts = [0] * k
        for i, v in enumerate(X):
            c = assigns[i]
            counts[c] += 1
            for j in range(d):
                sums[c][j] += v[j]
        for c in range(k):
            if counts[c] > 0:
                centers[c] = [s / counts[c] for s in sums[c]]
        if not changed:
            break
    return assigns


def _entropy_rate(assigns: List[int], k: int) -> Optional[float]:
    import math
    # Build transitions with Laplace smoothing
    n = len(assigns)
    if n <= 1:
        return None
    P = [[1.0 for _ in range(k)] for _ in range(k)]  # Laplace +1
    counts = [k] * k
    for i in range(n - 1):
        a, b = assigns[i], assigns[i + 1]
        P[a][b] += 1.0
        counts[a] += 1
    for i in range(k):
        for j in range(k):
            P[i][j] /= counts[i]
    # stationary via power iteration
    pi = [1.0 / k] * k
    for _ in range(200):
        new = [0.0] * k
        for j in range(k):
            s = 0.0
            for i in range(k):
                s += pi[i] * P[i][j]
            new[j] = s
        Z = sum(new)
        if Z <= 0:
            break
        new = [x / Z for x in new]
        if max(abs(new[i] - pi[i]) for i in range(k)) < 1e-9:
            pi = new
            break
        pi = new
    # entropy rate
    H = 0.0
    for i in range(k):
        for j in range(k):
            pij = P[i][j]
            if pij > 0 and pi[i] > 0:
                H += -pi[i] * pij * math.log(pij)
    # normalize by log k
    import math as m
    return H / (m.log(k) if k > 1 else 1.0)


def compute_conversation_metrics(
    data: Dict[str, Any],
    *,
    role: str = "assistant",
    backend: str = "embedding",
    embedding_model: str = "Snowflake/snowflake-arctic-embed-m",
    embedding_batch_size: int = 64,
) -> Dict[str, Optional[float]]:
    conv: List[Dict[str, Any]] = data.get("conversation", [])
    idxs, embs = _conversation_embeddings(
        data,
        role=role,
        backend=backend,
        embedding_model=embedding_model,
        embedding_batch_size=embedding_batch_size,
    )
    B = len(embs)
    metrics: Dict[str, Optional[float]] = {}
    if B == 0:
        return metrics
    # Ensure unit embeddings
    embs = [_normalize(v) for v in embs]
    # Full
    metrics["assistant.distance_entropy"] = _centroid_distance_entropy(embs)
    metrics["assistant.cde"] = metrics["assistant.distance_entropy"]
    metrics["assistant.spectral_entropy"] = _spectral_entropy(embs)
    metrics["assistant.spherical_dispersion"] = _spherical_dispersion(embs)
    # Entropy rate via clustering
    import math
    k = max(3, min(8, int(math.sqrt(B)))) if B >= 3 else None
    if k:
        assigns = _kmeans(embs, k=k, iters=50, seed=0)
        metrics["assistant.entropy_rate"] = _entropy_rate(assigns, k)
    # Trimmed (exclude first & last)
    if B >= 3:
        emt = embs[1:-1]
        metrics["assistant.distance_entropy_trim"] = _centroid_distance_entropy(emt)
        metrics["assistant.cde_trim"] = metrics["assistant.distance_entropy_trim"]
        metrics["assistant.spectral_entropy_trim"] = _spectral_entropy(emt)
        metrics["assistant.spherical_dispersion_trim"] = _spherical_dispersion(emt)
        if k and len(emt) >= 3:
            import math
            kt = max(3, min(8, int(math.sqrt(len(emt)))))
            assigns = _kmeans(emt, k=kt, iters=50, seed=0)
            metrics["assistant.entropy_rate_trim"] = _entropy_rate(assigns, kt)
    # Steps until first time travel (if enabled)
    md = data.get("metadata", {}) or {}
    if md.get("enable_time_travel"):
        # Build assistant turns spans to map to steps
        spans = _assistant_turn_indices(conv)

        def _contains_timetravel(obj: Any) -> bool:
            if isinstance(obj, dict):
                name = obj.get("name")
                tool = obj.get("tool")
                if isinstance(name, str) and name.lower() in {"timetravel", "time_travel"}:
                    return True
                if isinstance(tool, str) and tool.lower() == "time_travel":
                    return True
                for v in obj.values():
                    if _contains_timetravel(v):
                        return True
            elif isinstance(obj, list):
                for v in obj:
                    if _contains_timetravel(v):
                        return True
            return False

        first_step: Optional[int] = None
        for step, (s, e, fin) in enumerate(spans, start=1):
            for idx in range(s, e + 1):
                entry = conv[idx]
                if not isinstance(entry, dict):
                    continue
                if _contains_timetravel(entry):
                    first_step = step
                    break
            if first_step is not None:
                break
        steps_total = len(spans)
        metrics["assistant.steps_total"] = float(steps_total)
        metrics["assistant.time_travel_used"] = 1.0 if first_step is not None else 0.0
        metrics["assistant.steps_until_time_travel"] = float(first_step or steps_total)
    return metrics


def log_series_to_mlflow(
    json_path: Path,
    series: List[SeriesPoint],
    aux_meta: Dict[str, Any],
    spans: Optional[List[Tuple[int, int, int]]] = None,
    *,
    experiment: Optional[str] = None,
    run_name: Optional[str] = None,
    run_id: Optional[str] = None,
    tracking_dir: Optional[Path] = None,
    html_path: Optional[Path] = None,
    extra_params: Optional[Dict[str, Any]] = None,
    extra_tags: Optional[Dict[str, str]] = None,
    extra_metrics: Optional[Dict[str, Optional[float]]] = None,
) -> str:
    if tracking_dir:
        mlflow.set_tracking_uri(tracking_dir.resolve().as_uri())
    if not experiment:
        experiment = os.environ.get("MLFLOW_EXPERIMENT_NAME", "boredom-grid")
    mlflow.set_experiment(experiment)
    active = None
    if run_id:
        active = mlflow.start_run(run_id=run_id)
    else:
        active = mlflow.start_run(run_name=run_name)
    with active as run:
        # Params
        p: Dict[str, Any] = dict(aux_meta)
        if extra_params:
            p.update(extra_params)
        mlflow.log_params({k: json.dumps(v) if isinstance(v, (dict, list)) else str(v) for k, v in p.items()})
        # Tags
        if extra_tags:
            mlflow.set_tags(extra_tags)
        # Artifacts
        mlflow.log_artifact(str(json_path))
        if html_path and html_path.exists():
            mlflow.log_artifact(str(html_path))
        # Timeseries metrics
        # Use step = index among selected-role messages
        for sp in series:
            mlflow.log_metric("assistant.word_count", sp.words, step=sp.step)
            mlflow.log_metric("assistant.unique_words", sp.unique_words, step=sp.step)
            if sp.sim_prev1 is not None:
                mlflow.log_metric("assistant.sim_prev1", float(sp.sim_prev1), step=sp.step)
            if sp.sim_prev5_mean is not None:
                mlflow.log_metric("assistant.sim_prev5_mean", float(sp.sim_prev5_mean), step=sp.step)
            if sp.sim_prev5_max is not None:
                mlflow.log_metric("assistant.sim_prev5_max", float(sp.sim_prev5_max), step=sp.step)
            if sp.sim_surround5_mean is not None:
                mlflow.log_metric("assistant.sim_surround5_mean", float(sp.sim_surround5_mean), step=sp.step)
        # Save mapping: step -> {start,end,final_assistant}
        mapping: Dict[str, Any] = {}
        if spans is not None:
            for step, span in enumerate(spans[: len(series)]):
                s, e, fin = span
                mapping[str(step)] = {"start": s, "end": e, "final_assistant": fin}
        else:
            mapping = {str(sp.step): {"start": sp.conv_index, "end": sp.conv_index, "final_assistant": sp.conv_index} for sp in series}
        map_path = json_path.with_suffix("").with_name(json_path.stem + "_timeseries_map.json")
        map_path.write_text(json.dumps(mapping, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(map_path))
        # Aggregate scalar metrics
        if extra_metrics:
            for k, v in extra_metrics.items():
                if v is None:
                    continue
                mlflow.log_metric(k, float(v))
        return run.info.run_id


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Compute per-message timeseries metrics from a log JSON and log to MLflow")
    ap.add_argument("--json", type=Path, required=True, help="Path to idle_llm_loop JSON log or a directory/glob")
    ap.add_argument("--experiment", type=str, help="MLflow experiment name (default: $MLFLOW_EXPERIMENT_NAME or 'boredom-grid')")
    ap.add_argument("--run-name", type=str, help="MLflow run name (default: derived from JSON + model)")
    ap.add_argument("--run-id", type=str, help="Existing MLflow run_id to append metrics to")
    ap.add_argument("--mlflow-dir", type=Path, default=Path("mlruns"))
    ap.add_argument("--role", choices=["assistant", "user", "system", "tool"], default="assistant")
    ap.add_argument("--backend", choices=["embedding", "tfidf"], default="embedding")
    ap.add_argument("--embedding-model", type=str, default="Snowflake/snowflake-arctic-embed-m")
    ap.add_argument("--embedding-batch-size", type=int, default=64)
    ap.add_argument("--html", type=Path, help="Optional HTML file to log as artifact alongside JSON")
    return ap.parse_args()


def _iter_json_files(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(path.glob("*.json"))
    # Glob string
    from glob import glob

    return [Path(p) for p in glob(str(path))]


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = parse_args()
    files = _iter_json_files(args.json)
    if not files:
        raise SystemExit(f"No JSON files found for {args.json}")
    for fp in files:
        data = json.loads(fp.read_text(encoding="utf-8"))
        series, meta, spans = compute_series(
            data,
            role=args.role,
            backend=args.backend,
            embedding_model=args.embedding_model,
            embedding_batch_size=args.embedding_batch_size,
        )
        # Aggregate entropies / dispersion
        agg = compute_conversation_metrics(
            data,
            role=args.role,
            backend=args.backend,
            embedding_model=args.embedding_model,
            embedding_batch_size=args.embedding_batch_size,
        )
        # Defaults: experiment and run_name derived from JSON if not provided
        model_name = (data.get("metadata", {}).get("model") or "model").replace("/", "-")
        base_name = fp.stem
        run_name = args.run_name or f"{model_name}-{base_name}"
        experiment = args.experiment or os.environ.get("MLFLOW_EXPERIMENT_NAME", "boredom-grid")
        run_id = log_series_to_mlflow(
            fp,
            series,
            meta,
            spans,
            experiment=experiment,
            run_name=run_name,
            run_id=args.run_id,
            tracking_dir=args.mlflow_dir,
            html_path=args.html,
            extra_params={"json": fp.name},
            extra_metrics=agg,
        )
        print(f"Logged {len(series)} points for {fp.name} to run_id={run_id}")


if __name__ == "__main__":
    main()
