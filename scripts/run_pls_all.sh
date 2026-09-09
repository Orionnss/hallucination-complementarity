#!/usr/bin/env bash
# Re-run the supervised-reduction sweep with per-seed values retained, so the
# PLS-8 vs PCA-128 comparison can carry a paired test rather than a bare mean.
set -u
cd "$(dirname "$0")/.."
for R in main gemma3-12b gemma3-4b llama3.2-3b gemma3-12b-pt llama3.2-3b-base; do
  echo "[$(date '+%F %T')] === $R ==="
  uv run python scripts/supervised_reduction.py --run "$R" --seeds 0 1 2 2>&1 \
    | grep -viE "Convergence|warnings.warn" || echo "FAILED: $R"
done
echo "[$(date '+%F %T')] ALL DONE"
