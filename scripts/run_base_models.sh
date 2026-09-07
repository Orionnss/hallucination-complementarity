#!/usr/bin/env bash
# Full pipeline for the two BASE (non-instruction-tuned) generators, on one GPU.
#
#   llama3.2-3b-base   meta-llama/Llama-3.2-3B    ~1.5 s/item on triviaqa
#   gemma3-12b-pt      google/gemma-3-12b-pt      ~3.5 s/item on triviaqa
#
# Ordered by resource, not by model, so the GPU is released as early as possible: all
# GPU work except CHARM runs first, then the CPU-only stages, then CHARM last.
#
#   phase 1  stage 1, both models      GPU   generation + attention traces   ~24 h
#   phase 2  stage 2, both models      GPU   3 judges, 4-bit                 ~2 h
#   phase 3  stages 3-5, both models   CPU   probes, kappa/McNemar, post-hoc ~5 h
#   phase 4  CHARM, both models        GPU   most costly, so last
#
# After phase 2 the GPU is free until CHARM: phase 3 can be interrupted, moved to another
# machine, or rerun from disk without repeating anything expensive.
#
# Llama precedes Gemma within each phase: it is less than half the cost, so a problem with
# the base-model setup surfaces after ~7 h rather than ~24.
#
# The gates from run_downstream.sh are kept inline, because splitting its stages across
# phases means they can no longer be enforced by that script: judging only starts if
# stage 1 completed, and training only starts if the labels are non-degenerate. Without
# them a truncated extraction would produce plausible-looking but meaningless numbers.
#
# Base models have no chat template, so stage 1 switches to the few-shot completion prompt
# in prompts.py automatically (HFGenerator.completion_mode). The shots are fixed, drawn
# from outside the four evaluation datasets, and one abstains so the INVALID class still
# fires — verified on both models: answers come back 2-6 tokens, 0% truncated, with
# "I don't know" appearing at a usable rate.
#
# Everything is checkpointed per item, so this can be killed and relaunched at any point
# and will resume. Run it under nohup or tmux; it is a multi-day job.
#
# Usage:  nohup bash scripts/run_base_models.sh cuda:1 > runs_base.log 2>&1 &
#         tail -f runs_base.log

set -uo pipefail

DEVICE="${1:-cuda:1}"
N_JOBS="${2:-16}"
CFG=/dev/null
cd "$(dirname "$0")/.."

RUNS=(llama3.2-3b-base gemma3-12b-pt)
model_of() {
    case "$1" in
        llama3.2-3b-base) echo meta-llama/Llama-3.2-3B ;;
        gemma3-12b-pt)    echo google/gemma-3-12b-pt ;;
    esac
}
log() { echo "[$(date '+%F %T')] $*"; }

stage1_ok() {
    uv run python - "$1" <<'PY'
import json, sys
from pathlib import Path
run, ok = sys.argv[1], True
for ds in ["triviaqa", "nq_open", "squad_v2", "coqa"]:
    p = Path(f"runs/{run}/stage1_extract/{ds}/manifest.json")
    if not p.exists():
        print(f"  {ds}: MISSING"); ok = False; continue
    d = json.load(open(p))
    print(f"  {ds}: extracted={d['n_extracted']}/{d['pool_size']} failed={d['n_failed']}")
    if d["n_extracted"] < 0.95 * d["pool_size"]:
        print(f"  {ds}: INCOMPLETE"); ok = False
sys.exit(0 if ok else 1)
PY
}

labels_ok() {
    uv run python - "$1" <<'PY'
import json, sys
from pathlib import Path
run, ok = sys.argv[1], True
for ds in ["triviaqa", "nq_open", "squad_v2", "coqa"]:
    p = Path(f"runs/{run}/stage2_judge/{ds}/labels.json")
    if not p.exists():
        print(f"  {ds}: MISSING labels.json"); ok = False; continue
    d = json.load(open(p))
    rate = d.get("hallucination_rate_scored")
    print(f"  {ds}: scored={d['n_scored']} dropped={d['n_dropped_invalid']} "
          f"halluc_rate={rate} unanimous={d['agreement']['unanimous_rate']}")
    if d["n_scored"] < 500 or rate is None or not (0.02 < rate < 0.98):
        print(f"  {ds}: UNUSABLE"); ok = False
sys.exit(0 if ok else 1)
PY
}

# --- phase 1: generation + feature extraction, both models (GPU) -----------------
for RUN_ID in "${RUNS[@]}"; do
    log "=== PHASE 1 | $RUN_ID | stage 1 (GPU) ==="
    uv run python -m halluc.pipeline.stage1_extract \
        --model "$(model_of "$RUN_ID")" --run-id "$RUN_ID" --device "$DEVICE" \
        || log "$RUN_ID stage 1 FAILED (continuing; rerun to resume)"
done
log "=== PHASE 1 complete: both extractions banked ==="

# --- phase 2: judging, both models (GPU) ----------------------------------------
for RUN_ID in "${RUNS[@]}"; do
    log "=== PHASE 2 | $RUN_ID | stage 2 (GPU) ==="
    if ! stage1_ok "$RUN_ID"; then
        log "$RUN_ID stage 1 incomplete; skipping its judging"; continue
    fi
    uv run python -m halluc.pipeline.stage2_judge \
        --run-id "$RUN_ID" --device "$DEVICE" --config "$CFG" \
        || log "$RUN_ID stage 2 FAILED (continuing)"
done
log "=== PHASE 2 complete: GPU is now free until CHARM ==="

# --- phase 3: probes and analysis, both models (CPU) ----------------------------
for RUN_ID in "${RUNS[@]}"; do
    log "=== PHASE 3 | $RUN_ID | stages 3-5 (CPU) ==="
    if ! labels_ok "$RUN_ID"; then
        log "$RUN_ID labels failed validation; skipping its training"; continue
    fi
    uv run python -m halluc.pipeline.stage3_train \
        --run-id "$RUN_ID" --n-jobs "$N_JOBS" --config "$CFG" \
        || { log "$RUN_ID stage 3 FAILED"; continue; }
    uv run python -m halluc.pipeline.stage4_analysis --run-id "$RUN_ID" --config "$CFG" \
        || log "$RUN_ID stage 4 FAILED"
    uv run python -m halluc.pipeline.stage5_posthoc --run-id "$RUN_ID" --config "$CFG" \
        || log "$RUN_ID stage 5 FAILED"
done
log "=== PHASE 3 complete ==="

# --- phase 4: CHARM, most costly, needs stage 2's labels ------------------------
for RUN_ID in "${RUNS[@]}"; do
    log "=== PHASE 4 | $RUN_ID | CHARM (GPU) ==="
    # CHARM re-extracts attention itself, so the generator is passed explicitly rather
    # than inferred from the run directory.
    uv run python -m halluc.pipeline.stage6_charm \
        --run-id "$RUN_ID" --model "$(model_of "$RUN_ID")" \
        --device "$DEVICE" --scope pooled \
        || log "$RUN_ID CHARM FAILED (continuing)"
done

log "ALL DONE"
