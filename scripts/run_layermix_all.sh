#!/usr/bin/env bash
# LayerMix over every generator at stride 1. Stride matters: the smoke test ran at
# stride 4, which makes the paper's "contiguous band" claim untestable by construction
# (no two candidates are adjacent). Only stride 1 can confirm or refute it.
# CPU only; nohup'd because harness background jobs are reaped on this host.
set -u
cd "$(dirname "$0")/.."
for run in main gemma3-12b llama3.2-3b gemma3-4b gemma3-12b-pt llama3.2-3b-base; do
  echo "[$(date '+%F %T')] === $run ==="
  uv run python scripts/layermix.py --runs "$run" --seeds 0 1 2 --stride 1 \
    || echo "[$(date '+%F %T')] FAILED: $run"
done
echo "[$(date '+%F %T')] ALL DONE"
