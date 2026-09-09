#!/usr/bin/env bash
# Generalise the low-dimensionality result from Qwen to every generator.
#
# Two questions, six models, 3 seeds each, CPU only:
#   supervised_reduction  does PLS in a handful of components match PCA-128?
#   pls_traceback         how few raw hidden dimensions carry the signal, and are they
#                         the same ones every fold?
#
# Both write per-run files, so runs never overwrite each other and a crash costs only the
# model it was on. Nothing here touches the GPU, so it can run alongside CHARM.
set -uo pipefail
cd "$(dirname "$0")/.."
RUNS=(main gemma3-12b gemma3-4b llama3.2-3b llama3.2-3b-base gemma3-12b-pt)
log() { echo "[$(date '+%F %T')] $*"; }

for R in "${RUNS[@]}"; do
    log "=== $R | supervised_reduction ==="
    uv run python scripts/supervised_reduction.py --run "$R" --seeds 0 1 2 \
        2>&1 | grep -viE "Convergence|warnings.warn" || log "$R supervised_reduction FAILED"
    log "=== $R | pls_traceback ==="
    uv run python scripts/pls_traceback.py --run "$R" --seeds 0 1 2 \
        2>&1 | grep -viE "Convergence|warnings.warn" || log "$R pls_traceback FAILED"
done
log "ALL DONE"
