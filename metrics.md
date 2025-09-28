# Conversation “Interestingness” Metrics

This repo computes per–assistant-turn metrics using embeddings of the final assistant text in each turn. A “turn” is defined as contiguous reasoning + function/tool outputs leading up to the next assistant text message (we split on assistant text output).

All scalar metrics here are normalized to [0,1] where possible, so they are comparable across runs and easy to track in MLflow.

## Per‑Turn Time Series (logged with step = turn index)
- `assistant.word_count`: tokenized word count of the final assistant text (simple alnum, lowercase, len≥2).
- `assistant.unique_words`: number of unique tokens in that turn.
- `assistant.sim_prev1`: cosine similarity to the previous turn’s embedding (omitted for the first turn).
- `assistant.sim_prev5_mean`: mean cosine similarity to up to 5 previous turns.
- `assistant.sim_prev5_max`: max cosine similarity to up to 5 previous turns.
- `assistant.sim_surround5_mean`: mean cosine similarity to up to 5 previous and up to 5 next turns (temporal “context”).

Embeddings default to Snowflake `snowflake-arctic-embed-m` (SentenceTransformers) and are L2‑normalized. You can switch to a TF‑IDF backend.

## Scalar “Interestingness” Scores (per conversation)
These are computed over unit embeddings of all assistant turns (and a trimmed variant that excludes the first and last turns).

- `assistant.distance_entropy` (aka Centroid Distance Entropy, CDE)
  - Build the centroid direction ĉ by averaging all turn embeddings and normalizing.
  - For each turn i, compute a scalar distance dᵢ = max(0, 1 − cos(eᵢ, ĉ)). Larger values mean farther from the center.
  - Convert distances into probabilities pᵢ ∝ dᵢ + ε (ε>0 avoids degeneracy). Entropy H = −Σ pᵢ log pᵢ, normalized by log B (B=turns), yielding a value in [0,1].
  - Intuition: low when turns cluster near a common theme (collapsed); high when turns spread broadly.

- `assistant.spectral_entropy`
  - Compute the covariance of turn embeddings and its eigenvalues λⱼ ≥ 0. Normalize qⱼ = λⱼ / Σλⱼ.
  - Spectral entropy H = −Σ qⱼ log qⱼ, normalized by log d (d=embedding dimension).
  - Intuition: measures dimensional spread. Low if conversation lives on a narrow manifold (e.g., a loop), high if it explores many independent directions.

- `assistant.spherical_dispersion`
  - Resultant length R = ||Σ eᵢ|| / B on the unit sphere. Define dispersion SD = 1 − R in [0,1].
  - Intuition: low when turns point roughly the same way; high when many directions cancel out.

- `assistant.entropy_rate`
  - Cluster turns into k states (k ≈ min(8, max(3, ⌊√B⌋))) with k‑means on embeddings.
  - Estimate a Markov transition matrix P with Laplace smoothing and its stationary distribution π by power iteration.
  - Entropy rate H = −Σᵢ πᵢ Σⱼ Pᵢⱼ log Pᵢⱼ, normalized by log k.
  - Intuition: captures unpredictability over time; loops have low entropy rate even if two “poles” exist.

### Trimmed Variants
We also log `_trim` metrics (e.g., `assistant.cde_trim`) computed on turns 2…B−1 to avoid intro/outro bias.

### Time Travel Usage Metric
When the run enables the time travel tool, we log:

- `assistant.steps_until_time_travel`: number of assistant turns until the first time‑travel invocation. If it never occurs, this equals the total number of assistant turns.
- `assistant.steps_total`: total assistant turns in the conversation.
- `assistant.time_travel_used`: 1 if time travel was used at least once, else 0.

Detection scans the conversation in order and checks, per turn span, for a function/tool call named `timeTravel` (or canonical `time_travel`).

## Backend and Reproducibility
- Default backend is embeddings (`Snowflake/snowflake-arctic-embed-m`).
- You can switch to TF‑IDF in `analyze_log_mlflow.py` via `--backend tfidf`.
- All metrics are purely unsupervised and conversation‑relative.

## Why Multiple Scores?
A single center-distance entropy can be fooled by bi‑modal loops (two opposing directions). Spectral entropy and spherical dispersion see overall shape; entropy rate sees temporal structure. Together they give a robust picture of “interestingness.”
