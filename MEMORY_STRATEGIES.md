# Memory Strategies for Experiment Grid

**Problem:** Running experiments accumulates memory → OOM crashes

**Current mitigations already in runner:**
1. `gc.collect()` between runs
2. `BOREDOM_TS_DISABLE=1` — skips MLflow time series (embedding accumulation)
3. `token_limit` plugin — caps output to 2000 tokens per turn
4. Subprocess isolation — each experiment runs in its own process

---

## Additional Strategies

### 1. Embedding Model Caching (Already Implemented)

The collapse_detection module loads embeddings once. We're using `tfidf` backend for fast local analysis, which avoids the sentence-transformers OOM.

### 2. Log Streaming Instead of Accumulation

Current: Entire conversation stored in memory, written at end.
Better: Stream each turn to disk as it happens.

**Implementation:** Modify `idle_llm_loop.py` to write JSONL incrementally:
```python
# In run_loop, after each iteration:
with open(log_path.with_suffix('.jsonl'), 'a') as f:
    f.write(json.dumps({"turn": state.iteration, "output": output_item}) + '\n')
```

### 3. Embedding Lazy Loading

For collapse detection during experiments, we can:
- Skip real-time collapse detection (just log raw data)
- Run collapse analysis as a separate post-processing step
- This is already partially done via `BOREDOM_TS_DISABLE`

### 4. Process Pool vs Thread Pool

Current `run_grid.py` uses ThreadPoolExecutor. For memory isolation, ProcessPoolExecutor is better — each worker has its own memory space that's freed on exit.

However, for serial execution with rate limiting, we run one at a time anyway.

### 5. Ulimit Memory Cap

Can set memory limits per process:
```bash
ulimit -v 2000000  # 2GB virtual memory limit
python run_scaffold_grid.py
```

This causes the process to fail cleanly instead of OOM-killing.

### 6. Watchdog Script

A separate monitoring script that:
- Watches memory usage
- Kills runaway processes before OOM
- Restarts the grid runner

```bash
#!/bin/bash
# watchdog.sh
while true; do
    MEM=$(free | grep Mem | awk '{print $3/$2 * 100.0}')
    if (( $(echo "$MEM > 85" | bc -l) )); then
        echo "Memory > 85%, killing python processes"
        pkill -f "idle_llm_loop.py"
        sleep 10
    fi
    sleep 5
done
```

### 7. Minimal Grid Variant

For testing on constrained systems, run with:
- `--scaffold 1 2` (baseline + values only)
- `--model 1.5b` (smallest model)
- Single run per cell

This establishes whether the framework works before committing to full grid.

---

## Recommended Approach

For Tim's server (25GB disk, unknown RAM):

1. **Use existing runner as-is** — it has subprocess isolation + GC
2. **Run in tmux/screen** — survives SSH disconnection
3. **State file resumes** — if it crashes, just restart
4. **Start with Phase 1** — 1.5B only to validate framework

```bash
# Start in tmux
tmux new -s experiments

# Run Phase 1 only (1.5B models, all scaffolds)
cd ~/boredom
source .venv/bin/activate
python run_scaffold_grid.py --model 1.5b

# Detach: Ctrl+B then D
# Reattach: tmux attach -t experiments
```

If OOM occurs:
1. Restart the script — it resumes from state file
2. Check `results/scaffold_grid/grid_state.json` for progress
3. Consider reducing `--max-iterations` in runner if needed

---

*Created Jan 12, 2026 for scaffold-centric S5 experiments*
