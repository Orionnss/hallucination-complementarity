#!/usr/bin/env bash
# Cross-dataset transfer for SAPLMA+PCA+logreg: train on one dataset and on three,
# evaluate on the held-out ones. Pure sklearn on cached features, so this is CPU work --
# the generators go in parallel (6 BLAS threads each) rather than onto a GPU.
set -u
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6
export CUDA_VISIBLE_DEVICES=""

for R in main gemma3-12b gemma3-4b llama3.2-3b gemma3-12b-pt llama3.2-3b-base; do
  ( echo "[$(date '+%F %T')] === $R start ==="
    uv run python scripts/dataset_transfer.py --run "$R" --seeds 0 1 2 3 4 2>&1 \
      | grep -viE "Convergence|warnings.warn" || echo "FAILED: $R"
    echo "[$(date '+%F %T')] === $R done ===" ) > "logs_transfer_$R.txt" 2>&1 &
done
wait

echo "[$(date '+%F %T')] === merging ==="
uv run python scripts/dataset_transfer.py --merge 2>&1 | grep -viE "Convergence|warnings.warn"
cat logs_transfer_*.txt
rm -f logs_transfer_*.txt
echo "[$(date '+%F %T')] ALL DONE"
