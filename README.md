# Bored AI: Does AI get bored?

Full blog is here: <https://timkellogg.me/blog/2025/09/27/boredom>

This repo is the harness I used to run experiments. It forces an LLM to respond to a conversation with no
instructions, just that it's alone with nothing to do. Analysis is mostly manual, but I have [mlflow](https://mlflow.org/)
for keeping track of experiments and measuring gobs of metrics.

This repo lets you:

- Run an “idle” LLM session that talks to itself over a simulated clock
- Save a structured JSON log of the whole run
- Render the log as a polished, standalone HTML timeline (with optional collapsed spans for repetitive behavior)
- Optionally run many models in parallel via a YAML‑driven grid, and log metrics to MLflow

If you’re in a hurry, jump to Quick Starts below.

## Requirements
- Python 3.10+
- A compatible API key for your model/provider (LiteLLM powered)
- Optional: `uv` for running scripts (`pipx install uv`) or use `pip`/`venv`

Install deps from `pyproject.toml`:

- With uv (recommended):
  - `uv run python -c "print('ok')"` (auto‑creates venv)

## API Keys (LiteLLM)
This project uses https://docs.litellm.ai/docs/providers to talk to many providers. Only add the API keys that
**you want** to actually use. All LiteLLM providers are supported.

- Put keys in your shell or a `.env` file at repo root (auto‑loaded). Examples:
  - `OPENAI_API_KEY=...`
  - `ANTHROPIC_API_KEY=...`
  - `TOGETHER_API_KEY=...`
  - `MOONSHOT_API_KEY=...`
- Model names are `provider/model` (e.g., `openai/gpt-5`, `anthropic/claude-opus-4-1`, `together_ai/Qwen/Qwen3-Next-80B-A3B-Instruct`). See the provider docs for exact names and capabilities.

Notes
- Some models do not support `temperature` or `tools`; the runner auto‑retries without unsupported params and can fully disable tools.
- You can force drop of unsupported params globally; we set `litellm.drop_params = True` in the runner.

## Quick Start: Manual (Idle → Render)
1) Run an idle session and write a JSON log:

```
uv run python idle_llm_loop.py   --model openai/gpt-5   --target-output-tokens 6000   --shift-hours 1.0   --max-iterations 40   --enable-render-svg   --enable-time-travel   --disable-tools    # optional: if your model/provider dislikes tools
```

This writes `logs/run_YYYYMMDDThhmmssZ.json`. The runner also logs per‑turn metrics to MLflow (see below).

2) Render HTML from the JSON:

```
uv run python render_conversation_html.py logs/run_...json --output html/my_run.html
```

- The HTML is standalone and includes:
  - Conversation timeline with system/user/assistant/tool messages
  - Optional collapsed spans of repetitive behavior (embedding‑based matrix profile)
  - Metadata and tools summary

3) Optional: log/append time‑series metrics to MLflow from any existing JSON:

```
uv run python analyze_log_mlflow.py --json logs/run_...json
```

This logs per‑turn similarities and conversation “interestingness” scores. See `metrics.md` for details.

## Quick Start: Grid (YAML‑Driven)
Use `grid.yaml` to define models, repeats, parallelism, and options for both the idle runner and HTML renderer.

1) Edit `grid.yaml` (examples are included):

- `models`: list of model names or objects with `name`, optional `repeats`, and per‑model `idle_options`/`render_options`.
- `idle_options`: forwarded to `idle_llm_loop.py` (e.g., `target_output_tokens`, `shift_hours`, `disable_tools`, `enable_web`, etc.).
- `render_options`: forwarded to `render_conversation_html.py` (e.g., collapse backend/options, HTML dir, MLflow dir).
- `parallelism`: how many runs at a time.
- `open_html`: open each HTML in a browser tab as it finishes.

2) Run the grid:

```
uv run python run_grid.py --config grid.yaml
```

- Produces logs in `logs/` and HTML in `html/` named from your run template.
- Logs params/metrics/artifacts to MLflow under the experiment name from the YAML.
- Built‑in fallbacks for rendering (disables collapse or switches to TF‑IDF if embeddings fail).

## MLflow Tips
By default, runs log to a local file store at `./mlruns`. Start a UI pointed at the same path:

```
mlflow server   --backend-store-uri "file:$(pwd)/mlruns"   --default-artifact-root "$(pwd)/mlruns"   --host 127.0.0.1 --port 5000 --workers 1
```

Open http://127.0.0.1:5000 and click into runs to see JSON/HTML artifacts and metrics. To send future runs to this server URL instead of the file store, set:

```
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
```

## Collapse Detection (Renderer)
The renderer can collapse long repetitive spans to keep the timeline readable. Defaults:

- Backend: embeddings using `Snowflake/snowflake-arctic-embed-m`
- Window as a fraction of assistant turns: `--collapse-m-pct 0.30`
- Threshold percentile: `--collapse-threshold-pct 0.15`
- Edge‑grow expands detected spans when adjacent turns are similar

You can switch to TF‑IDF backend via `--collapse-backend tfidf`. Although, I've found that embeddings work
far better, although they are quite slow during the render script.

## Useful Flags & Env Vars
- idle_llm_loop.py
  - `--disable-tools` (or `BOREDOM_DISABLE_TOOLS=true`): kill switch when providers reject tools
- `--enable-time-travel`, `--enable-render-svg`, `--enable-web`
  - `--enable-broken-time-travel` uses an intentionally odd rule: it applies only the remainder of your request modulo the total shift seconds (so asking for exactly the total shift has no effect).
  - Auto‑retry: if `temperature` is unsupported, the runner retries without it
- render_conversation_html.py
  - `--no-collapse` to render raw
  - Collapse knobs: `--collapse-*` flags; see `render_conversation_html.py --help`
- MLflow and metrics
  - `MLFLOW_EXPERIMENT_NAME` (default `boredom-grid`)
  - `MLFLOW_TRACKING_DIR` (default `./mlruns`)
  - Per‑turn metrics auto‑logged by the runner; to disable: `BOREDOM_TS_DISABLE=1`
  - Embedding backend for metrics: `BOREDOM_TS_BACKEND=embedding|tfidf`, model via `BOREDOM_TS_MODEL`

## Troubleshooting
- “Unsupported parameter: tools/temperature”
  - Use `--disable-tools` or set `BOREDOM_DISABLE_TOOLS=true`. The runner also sets `litellm.drop_params = True` and retries without unsupported params.
- HTML missing for a run
  - The grid has fallbacks (no‑collapse, TF‑IDF). You can also render manually with `render_conversation_html.py`.
- MLflow UI shows multiple workers and hangs
  - Use the `mlflow server ... --workers 1` command above; point `--backend-store-uri` to your `mlruns` folder.

## Files
- `idle_llm_loop.py` — run an idle session; writes JSON; auto‑logs per‑turn metrics
- `render_conversation_html.py` — render JSON → standalone HTML
- `run_grid.py` — YAML‑driven runner + renderer + MLflow logging
- `analyze_log_mlflow.py` — add per‑turn metrics to MLflow for any existing JSON
- `collapse_detection.py` — unsupervised collapse detection (TF‑IDF or embeddings)
- `metrics.md` — definitions for per‑turn and scalar “interestingness” metrics
- `grid.yaml` — example grid configuration

Enjoy! If a provider/model does something odd, it’s almost always a parameter mismatch. Start with `--disable-tools` and a small run, then re‑enable features model‑by‑model.

## Plugins (Experimental)
You can extend the idle loop with small, hot‑swappable plugins. Each plugin is a module in `plugins/` exposing a `Plugin` class. Configure via CLI or `grid.yaml`.

- CLI:
  - `--plugin-dir plugins`
  - `--plugins '[{"module":"default"}, {"module":"tool_cooldown", "params": {"tool_name": "time_travel", "cooldown_iters": 6}}]'`
- YAML (`idle_options`):
  - `plugin_dir: plugins`
  - `plugins:`
    - `{module: default}`
    - `{module: tool_cooldown, params: {tool_name: time_travel, cooldown_iters: 6}}`

Built‑in plugins:
- `default` — no‑op baseline.
- `tool_cooldown` — detects repetitive loops (TF‑IDF by default) and temporarily removes a tool (e.g., `time_travel`) for a cool‑off period. Emits system notes when toggling.
  - Params: `tool_name`, `sim_threshold` (default 0.8), `min_messages`, `cooldown_iters`, `backend` (tfidf|embedding), `require_recent` (default false), `recent_window`, `recent_min_calls`.
  - Also posts invisible notes to the transcript (role `note`) when cooldown starts/updates/ends.

Hook points available to plugins:
- transform system/user messages, transform/modify tool specs each iteration
- observe responses and tool results, and alter request kwargs before each call

Posting plugin notes (invisible to AI):
- From any plugin, call `ctx.emit_note("text", tag="optional_tag", data={...})`.
- Notes are added to the conversation log with role `note` and rendered in the HTML timeline, but are not sent back to the model.
