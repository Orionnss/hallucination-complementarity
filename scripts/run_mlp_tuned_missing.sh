#!/usr/bin/env bash
# Tuned-MLP SAPLMA for the three generators the original run never covered, so the
# detector table can separate the published MLP, the tuned MLP and PCA+logreg on every
# block rather than only on Qwen3-14B. Each generator writes its own file; they are
# merged at the end.
set -u
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=""

for R in gemma3-12b gemma3-4b llama3.2-3b; do
  ( echo "[$(date '+%F %T')] === $R start ==="
    uv run python scripts/saplma_mlp_tuned.py --runs "$R" --seeds 0 1 2 3 4 \
      --out "runs/mlp_tuned_parts/$R.json" 2>&1 \
      | grep -viE "Convergence|warnings.warn|Stochastic Optimizer" || echo "FAILED: $R"
    echo "[$(date '+%F %T')] === $R done ===" ) > "logs_mlp_$R.txt" 2>&1 &
done
wait

echo "[$(date '+%F %T')] === merging ==="
uv run python scripts/saplma_mlp_tuned.py --merge runs/mlp_tuned_parts/*.json
cat logs_mlp_*.txt
rm -f logs_mlp_*.txt
echo "[$(date '+%F %T')] ALL DONE"
