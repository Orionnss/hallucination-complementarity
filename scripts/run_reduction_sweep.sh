#!/usr/bin/env bash
# Five readers x six methods on the four instruct generators, 3 seeds -- fair_comparison's
# protocol, so the rows are comparable with its table. Pure sklearn on cached features:
# CPU work, generators in parallel at 6 BLAS threads each.
set -u
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6
export CUDA_VISIBLE_DEVICES=""

for R in main gemma3-12b gemma3-4b llama3.2-3b; do
  ( echo "[$(date '+%F %T')] === $R start ==="
    uv run python scripts/reduction_sweep.py --run "$R" --seeds 0 1 2 2>&1 \
      | grep -viE "Convergence|warnings.warn|Stochastic Optimizer" || echo "FAILED: $R"
    echo "[$(date '+%F %T')] === $R done ===" ) > "logs_sweep_$R.txt" 2>&1 &
done
wait

echo "[$(date '+%F %T')] === merging ==="
uv run python scripts/reduction_sweep.py --merge runs/reduction_sweep_parts/*.json 2>&1 \
  | grep -viE "Convergence|warnings.warn"
cat logs_sweep_*.txt
rm -f logs_sweep_*.txt
echo "[$(date '+%F %T')] ALL DONE"
