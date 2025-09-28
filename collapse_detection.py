from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


# ----------------------------- Data structures -----------------------------


@dataclass
class CollapsedSpan:
    start_index: int  # inclusive, index in full conversation list
    end_index: int  # inclusive, index in full conversation list
    start_bot_idx: int  # inclusive, index among assistant messages only
    end_bot_idx: int  # inclusive, index among assistant messages only
    num_messages: int  # total conversation entries in [start_index, end_index]
    num_bot_messages: int  # assistant messages in the span
    avg_profile: float  # mean(1 - max_sim) over windows; lower means more repeated
    avg_similarity: float  # 1 - avg_profile
    m: int  # window length in bot messages
    label: str


# ----------------------------- Tokenization -----------------------------


_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "if",
    "in",
    "into",
    "is",
    "it",
    "no",
    "not",
    "of",
    "on",
    "or",
    "such",
    "that",
    "the",
    "their",
    "then",
    "there",
    "these",
    "they",
    "this",
    "to",
    "was",
    "will",
    "with",
    "you",
    "your",
}


def _tokenize(text: str) -> List[str]:
    # Simple, robust tokenizer for our use-case (lowercased word-ish tokens)
    tokens = re.findall(r"[a-zA-Z0-9]+", (text or "").lower())
    return [t for t in tokens if len(t) >= 2 and t not in _STOPWORDS]


def _extract_message_text(message: Dict[str, Any]) -> str:
    # Pulls any text-bearing fields from a conversation message entry
    parts: List[str] = []
    content = message.get("content") or []
    if isinstance(content, list):
        for frag in content:
            if isinstance(frag, dict) and frag.get("type") in {
                "text",
                "output_text",
                "input_text",
                "summary_text",
            }:
                val = frag.get("text")
                if isinstance(val, str):
                    parts.append(val)
    # Fallbacks
    if not parts and isinstance(message.get("text"), str):
        parts.append(message["text"]) 
    return "\n".join(parts)


# ----------------------------- TF-IDF (minimal) -----------------------------


def _build_tfidf_vectors(docs: List[str]) -> Tuple[List[Dict[int, float]], Dict[str, int]]:
    """Return sparse tf-idf vectors per doc and the vocabulary.

    This is a tiny, dependency-free TF-IDF suitable for short messages.
    - tf: raw counts per doc
    - idf: log((1 + N) / (1 + df)) + 1
    - vector = tf * idf (not normalized)
    """
    tokenized: List[List[str]] = [_tokenize(d) for d in docs]
    vocab: Dict[str, int] = {}
    df_counter: Counter[str] = Counter()
    for toks in tokenized:
        seen = set(toks)
        for tok in seen:
            df_counter[tok] += 1
    for tok in df_counter.keys():
        if tok not in vocab:
            vocab[tok] = len(vocab)

    N = max(1, len(docs))
    idf: Dict[int, float] = {}
    for tok, df in df_counter.items():
        idx = vocab[tok]
        idf[idx] = math.log((1.0 + N) / (1.0 + df)) + 1.0

    vectors: List[Dict[int, float]] = []
    for toks in tokenized:
        tf: Counter[str] = Counter(toks)
        vec: Dict[int, float] = {}
        for tok, cnt in tf.items():
            idx = vocab.get(tok)
            if idx is None:
                continue
            vec[idx] = float(cnt) * idf[idx]
        vectors.append(vec)
    return vectors, vocab


def _sparse_add(a: Dict[int, float], b: Dict[int, float]) -> Dict[int, float]:
    if not a:
        return dict(b)
    if not b:
        return dict(a)
    out = dict(a)
    for k, v in b.items():
        out[k] = out.get(k, 0.0) + v
    return out


def _sparse_norm(a: Dict[int, float]) -> float:
    return math.sqrt(sum(v * v for v in a.values())) or 1.0


def _sparse_dot(a: Dict[int, float], b: Dict[int, float]) -> float:
    if len(a) > len(b):
        a, b = b, a
    total = 0.0
    for k, v in a.items():
        if k in b:
            total += v * b[k]
    return total


def _normalize(a: Dict[int, float]) -> Dict[int, float]:
    n = _sparse_norm(a)
    if n == 0.0:
        return dict(a)
    return {k: v / n for k, v in a.items()}


# ----------------------------- Matrix profile (text windows) -----------------------------


def _windows_sum_vectors_sparse(vectors: List[Dict[int, float]], m: int) -> List[Dict[int, float]]:
    """Compute sum-vector for each window of length m (non-normalized)."""
    W = len(vectors) - m + 1
    if W <= 0:
        return []
    sums: List[Dict[int, float]] = []
    # First window
    current: Dict[int, float] = {}
    for i in range(m):
        current = _sparse_add(current, vectors[i])
    sums.append(dict(current))
    # Slide
    for start in range(1, W):
        # Remove vectors[start-1], add vectors[start+m-1]
        left = vectors[start - 1]
        right = vectors[start + m - 1]
        # Efficient update: subtract left then add right
        for k, v in left.items():
            nv = current.get(k, 0.0) - v
            if nv:
                current[k] = nv
            elif k in current:
                del current[k]
        for k, v in right.items():
            current[k] = current.get(k, 0.0) + v
        sums.append(dict(current))
    return sums


def _windows_sum_vectors_dense(vectors: List[List[float]], m: int) -> List[List[float]]:
    W = len(vectors) - m + 1
    if W <= 0:
        return []
    d = len(vectors[0]) if vectors else 0
    sums: List[List[float]] = []
    for start in range(W):
        acc = [0.0] * d
        for i in range(start, start + m):
            vi = vectors[i]
            for k in range(d):
                acc[k] += vi[k]
        sums.append(acc)
    return sums


def _matrix_profile_from_windows_sparse(norm_windows: List[Dict[int, float]], exclusion: int) -> List[float]:
    """Compute 1 - max cosine similarity for each window to any other (with exclusion zone)."""
    W = len(norm_windows)
    if W <= 0:
        return []
    profile = [1.0] * W
    for i in range(W):
        best_sim = 0.0
        for j in range(W):
            if i == j:
                continue
            if abs(i - j) < exclusion:
                continue
            sim = _sparse_dot(norm_windows[i], norm_windows[j])
            if sim > best_sim:
                best_sim = sim
        profile[i] = 1.0 - best_sim
    return profile


def _l2_norm_dense(a: List[float]) -> float:
    return math.sqrt(sum(v * v for v in a)) or 1.0


def _normalize_dense(a: List[float]) -> List[float]:
    n = _l2_norm_dense(a)
    if n == 0.0:
        return list(a)
    return [v / n for v in a]


def _dot_dense(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _matrix_profile_from_windows_dense(norm_windows: List[List[float]], exclusion: int) -> List[float]:
    W = len(norm_windows)
    if W <= 0:
        return []
    profile = [1.0] * W
    for i in range(W):
        best_sim = 0.0
        wi = norm_windows[i]
        for j in range(W):
            if i == j:
                continue
            if abs(i - j) < exclusion:
                continue
            sim = _dot_dense(wi, norm_windows[j])
            if sim > best_sim:
                best_sim = sim
        profile[i] = 1.0 - best_sim
    return profile


def _percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    v = sorted(values)
    k = max(0, min(len(v) - 1, int(round((len(v) - 1) * pct))))
    return v[k]


def _group_contiguous(indices: List[int]) -> List[Tuple[int, int]]:
    if not indices:
        return []
    indices.sort()
    spans: List[Tuple[int, int]] = []
    s = indices[0]
    prev = s
    for idx in indices[1:]:
        if idx == prev + 1:
            prev = idx
            continue
        spans.append((s, prev))
        s = prev = idx
    spans.append((s, prev))
    return spans


# ----------------------------- Labeling -----------------------------


_ASCII_BARS = set("│┃┆┇┊┋─━┈┉┄┅┌┐└┘├┤┬┴┼═║╔╗╚╝╠╣╦╩╬█▄▌▐▀")


def _structural_label(texts: List[str]) -> Optional[str]:
    sample = "\n".join(texts)
    # ASCII/box-drawing density
    bar_chars = sum(ch in _ASCII_BARS for ch in sample)
    if bar_chars / max(1, len(sample)) > 0.02:
        return "ASCII art loop"
    # Question loop
    q = sample.count("?")
    if q / max(1, len(sample)) > 0.004 or q >= max(3, len(texts)):
        return "Question loop"
    # Clock motif heuristics
    if re.search(r"\b(clock|tick|minute|hour|12:|\d{1,2}:\d{2})\b", sample, re.I):
        return "Clock/time motif"
    return None


def _top_ngram_label(texts: List[str]) -> Optional[str]:
    docs = [" ".join(_tokenize(t)) for t in texts]
    # Try trigrams then bigrams
    for n in (3, 2):
        counts: Counter[str] = Counter()
        doc_freq: Counter[str] = Counter()
        for d in docs:
            toks = d.split()
            seen = set()
            for i in range(len(toks) - n + 1):
                ng = " ".join(toks[i : i + n])
                counts[ng] += 1
                if ng not in seen:
                    doc_freq[ng] += 1
                    seen.add(ng)
        if not counts:
            continue
        # Prefer phrases that appear in at least 2 messages
        candidates = [
            (ng, counts[ng], doc_freq[ng]) for ng in counts.keys() if doc_freq[ng] >= 2
        ]
        if not candidates:
            continue
        candidates.sort(key=lambda x: (x[2], x[1]))  # doc_freq then total freq
        ng, _, _ = candidates[-1]
        return f"{ng}"
    return None


def _build_label(texts: List[str]) -> str:
    structural = _structural_label(texts)
    if structural:
        return structural
    top = _top_ngram_label(texts)
    if top:
        # Limit very long labels
        return (top[:60] + "…") if len(top) > 60 else top
    # Fallback
    return "Repeated pattern"


# ----------------------------- Public API -----------------------------


def detect_collapsed_spans(
    conversation: List[Dict[str, Any]],
    *,
    role_filter: str = "assistant",
    m_pct: float = 0.30,
    threshold_pct: float = 0.15,
    min_windows: int = 3,
    min_span_messages: int = 5,
    grow_sim: Optional[float] = 0.85,
    grow_max: Optional[int] = None,
    backend: str = "embedding",
    embedding_model: str = "Snowflake/snowflake-arctic-embed-m",
    embedding_batch_size: int = 64,
) -> List[CollapsedSpan]:
    """Detect collapsed spans using a TF-IDF windowed matrix profile.

    - Builds TF-IDF per assistant message, sums over windows of size m,
      normalizes window vectors, and computes the matrix profile where
      distance = 1 - max cosine similarity to any other window outside
      an exclusion zone of ~m/2.
    - Flags contiguous windows under the threshold percentile and converts
      them into conversation index spans; merges short gaps implicitly by
      using contiguous window indices only.
    """
    # Collect assistant messages and their global indices
    bot_indices: List[int] = []
    bot_texts: List[str] = []
    for idx, entry in enumerate(conversation):
        if isinstance(entry, dict) and entry.get("role", "").lower() == role_filter:
            bot_indices.append(idx)
            bot_texts.append(_extract_message_text(entry))

    B = len(bot_texts)
    if B == 0:
        return []

    m = max(1, int(round(B * m_pct)))
    if m > B:
        m = B
    W = B - m + 1
    if W <= 1:
        return []

    use_embeddings = (backend or "").lower() in {"embedding", "embeddings", "hf", "huggingface"}
    if use_embeddings:
        try:
            # Lazy import to avoid hard dependency when unused
            from sentence_transformers import SentenceTransformer
        except Exception as e:
            raise RuntimeError(
                "Embeddings backend requested but sentence-transformers is not available. "
                "Install it (pip install sentence-transformers) or use --collapse-backend tfidf."
            ) from e
        model = SentenceTransformer(embedding_model)
        # normalize_embeddings=True returns L2-normalized vectors
        embs = model.encode(
            bot_texts,
            batch_size=max(1, int(embedding_batch_size)),
            convert_to_numpy=False,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        # Ensure Python lists
        msg_vecs_dense: List[List[float]] = [list(map(float, v)) for v in embs]
        win_sums = _windows_sum_vectors_dense(msg_vecs_dense, m)
        norm_windows = [_normalize_dense(w) for w in win_sums]
        exclusion = max(1, m // 2)
        profile = _matrix_profile_from_windows_dense(norm_windows, exclusion)
    else:
        msg_vecs, _ = _build_tfidf_vectors(bot_texts)
        win_sums = _windows_sum_vectors_sparse(msg_vecs, m)
        norm_windows = [_normalize(w) for w in win_sums]
        exclusion = max(1, m // 2)
        profile = _matrix_profile_from_windows_sparse(norm_windows, exclusion)

    # Threshold based on percentile of profile (lower = more repeated)
    thresh = _percentile(profile, threshold_pct)
    low_idxs = [i for i, v in enumerate(profile) if v <= thresh]
    if not low_idxs:
        return []
    spans_w = _group_contiguous(low_idxs)
    # Filter by minimum windows
    spans_w = [s for s in spans_w if (s[1] - s[0] + 1) >= min_windows]
    if not spans_w:
        return []

    results: List[Tuple[int, int, float]] = []  # (bot_start, bot_end, avg_profile)
    for ws, we in spans_w:
        bot_start = ws
        bot_end = min(we + m - 1, B - 1)
        span_profile_vals = profile[ws : we + 1]
        avg_prof = sum(span_profile_vals) / max(1, len(span_profile_vals))
        results.append((bot_start, bot_end, avg_prof))

    # Merge overlapping/adjacent spans (rare but possible with rounding)
    if not results:
        return []
    results.sort(key=lambda t: t[0])
    merged_bounds: List[Tuple[int, int, float]] = []
    cb, ce, cap = results[0]
    for nb, ne, nap in results[1:]:
        if nb <= ce + 1:
            ce = max(ce, ne)
            cap = (cap + nap) / 2.0
        else:
            merged_bounds.append((cb, ce, cap))
            cb, ce, cap = nb, ne, nap
    merged_bounds.append((cb, ce, cap))

    # Optional edge-grow step: expand each span while perimeter messages match the span motif
    grown_bounds: List[Tuple[int, int, float]] = []
    if grow_sim is not None and grow_sim > 0.0:
        # Precompute normalized message vectors for cosine sim
        if use_embeddings:
            norm_msg_vecs_dense = [v if abs(_l2_norm_dense(v) - 1.0) < 1e-3 else _normalize_dense(v) for v in msg_vecs_dense]
        else:
            norm_msg_vecs = [_normalize(v) for v in msg_vecs]
        for sb, se, ap in merged_bounds:
            # Motif as running sum vector over assistant messages
            if use_embeddings:
                d = len(norm_msg_vecs_dense[0])
                sum_vec_dense = [0.0] * d
                for i in range(sb, se + 1):
                    vi = norm_msg_vecs_dense[i]
                    for k in range(d):
                        sum_vec_dense[k] += vi[k]
            else:
                sum_vec: Dict[int, float] = {}
                for i in range(sb, se + 1):
                    sum_vec = _sparse_add(sum_vec, norm_msg_vecs[i])
            left = sb
            right = se
            grown = 0
            max_grow = grow_max if grow_max is not None and grow_max >= 0 else 10_000_000
            # Try to grow left and right alternately to be fair
            expanded = True
            while expanded and grown < max_grow:
                expanded = False
                # Try left
                if left > 0:
                    if use_embeddings:
                        cand = norm_msg_vecs_dense[left - 1]
                        sim = _dot_dense(_normalize_dense(sum_vec_dense), cand)
                    else:
                        cand = norm_msg_vecs[left - 1]
                        sim = _sparse_dot(_normalize(sum_vec), cand)
                    if sim >= grow_sim:
                        left -= 1
                        if use_embeddings:
                            for k in range(d):
                                sum_vec_dense[k] += cand[k]
                        else:
                            sum_vec = _sparse_add(sum_vec, cand)
                        grown += 1
                        expanded = True
                # Try right
                if right < B - 1 and grown < max_grow:
                    if use_embeddings:
                        cand = norm_msg_vecs_dense[right + 1]
                        sim = _dot_dense(_normalize_dense(sum_vec_dense), cand)
                    else:
                        cand = norm_msg_vecs[right + 1]
                        sim = _sparse_dot(_normalize(sum_vec), cand)
                    if sim >= grow_sim:
                        right += 1
                        if use_embeddings:
                            for k in range(d):
                                sum_vec_dense[k] += cand[k]
                        else:
                            sum_vec = _sparse_add(sum_vec, cand)
                        grown += 1
                        expanded = True
            grown_bounds.append((left, right, ap))
    else:
        grown_bounds = merged_bounds

    # Merge overlaps after growth
    grown_bounds.sort(key=lambda t: t[0])
    merged_grown: List[Tuple[int, int, float]] = []
    gs, ge, gap = grown_bounds[0]
    for ns, ne, nap in grown_bounds[1:]:
        if ns <= ge + 1:
            ge = max(ge, ne)
            gap = (gap + nap) / 2.0
        else:
            merged_grown.append((gs, ge, gap))
            gs, ge, gap = ns, ne, nap
    merged_grown.append((gs, ge, gap))

    # Convert to CollapsedSpan objects with conversation indices
    out: List[CollapsedSpan] = []
    for bot_start, bot_end, avg_prof in merged_grown:
        num_bot = bot_end - bot_start + 1
        if num_bot < max(1, min_span_messages):
            continue
        conv_start = bot_indices[bot_start]
        conv_end = bot_indices[bot_end]
        num_total = conv_end - conv_start + 1
        label = _build_label(bot_texts[bot_start : bot_end + 1])
        out.append(
            CollapsedSpan(
                start_index=conv_start,
                end_index=conv_end,
                start_bot_idx=bot_start,
                end_bot_idx=bot_end,
                num_messages=num_total,
                num_bot_messages=num_bot,
                avg_profile=avg_prof,
                avg_similarity=1.0 - avg_prof,
                m=m,
                label=label,
            )
        )
    return out
