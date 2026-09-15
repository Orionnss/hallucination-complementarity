#!/usr/bin/env bash
# One PLS model per detector source, all predicting the same label, on the four instruct
# generators at 3 seeds. CPU-only sklearn; generators in parallel at 6 BLAS threads.
set -u
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6
export CUDA_VISIBLE_DEVICES=""
for R in main gemma3-12b gemma3-4b llama3.2-3b; do
  ( echo "[$(date '+%F %T')] === $R start ==="
    uv run python scripts/pls_relevance.py --run "$R" --seeds 0 1 2 2>&1 \
      | grep -viE "Convergence|warnings.warn|y residual" || echo "FAILED: $R"
    echo "[$(date '+%F %T')] === $R done ===" ) > "logs_rel_$R.txt" 2>&1 &
done
wait
echo "[$(date '+%F %T')] === merging ==="
uv run python scripts/pls_relevance.py --merge runs/pls_relevance_parts/*.json 2>&1 \
  | grep -viE "Convergence|warnings.warn|y residual"
cat logs_rel_*.txt; rm -f logs_rel_*.txt
echo "[$(date '+%F %T')] ALL DONE"
