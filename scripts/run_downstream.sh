#!/usr/bin/env bash
# Wait for one generator's stage 1 to finish, then run stages 2-5 for that run.
#
# Every stage is pinned to the same --run-id, so a second generator can never write
# into another's directory. Stage 3 is gated on stage 2 having produced usable labels:
# training on truncated or one-class labels would yield meaningless numbers rather than
# failing loudly.
#
# Usage: scripts/run_downstream.sh <run-id> <judge-device> [n_jobs]

set -uo pipefail

RUN_ID="${1:?usage: run_downstream.sh <run-id> <judge-device> [n_jobs]}"
DEVICE="${2:?usage: run_downstream.sh <run-id> <judge-device> [n_jobs]}"
N_JOBS="${3:-16}"
CFG=/dev/null
log() { echo "[$(date '+%F %T')] [$RUN_ID] $*"; }

# --- wait for this run's stage 1 -------------------------------------------------
PATTERN="stage1_extract.*--run-id $RUN_ID"
if pgrep -f "$PATTERN" >/dev/null; then
    log "waiting for stage 1"
    while pgrep -f "$PATTERN" >/dev/null; do sleep 60; done
fi
log "stage 1 not running; checking it completed"
if ! uv run python - "$RUN_ID" <<'PY'
import json, sys
from pathlib import Path
run = sys.argv[1]
ok = True
for ds in ["triviaqa", "nq_open", "squad_v2", "coqa"]:
    p = Path(f"runs/{run}/stage1_extract/{ds}/manifest.json")
    if not p.exists():
        print(f"  {ds}: MISSING"); ok = False; continue
    d = json.load(open(p))
    print(f"  {ds}: extracted={d['n_extracted']}/{d['pool_size']} failed={d['n_failed']} "
          f"generator={d['generator']}")
    if d["n_extracted"] < 0.95 * d["pool_size"]:
        print(f"  {ds}: INCOMPLETE"); ok = False
sys.exit(0 if ok else 1)
PY
then
    log "stage 1 incomplete; stopping"; exit 1
fi

# --- stage 2: judges ------------------------------------------------------------
log "stage 2 (judge pool, batched, $DEVICE)"
uv run python -m halluc.pipeline.stage2_judge --run-id "$RUN_ID" --device "$DEVICE" --config "$CFG" \
    || { log "stage 2 failed"; exit 1; }

# Gate: labels must exist for every dataset with a non-degenerate class balance.
if ! uv run python - "$RUN_ID" <<'PY'
import json, sys
from pathlib import Path
run = sys.argv[1]
ok = True
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
then
    log "stage 2 labels failed validation; not starting stage 3"; exit 1
fi

# --- stages 3-5 -----------------------------------------------------------------
log "stage 3 (pooled training, per-dataset eval, n_jobs=$N_JOBS)"
uv run python -m halluc.pipeline.stage3_train --run-id "$RUN_ID" --n-jobs "$N_JOBS" --config "$CFG" \
    || { log "stage 3 failed"; exit 1; }

log "stage 4 (complementarity: kappa, McNemar)"
uv run python -m halluc.pipeline.stage4_analysis --run-id "$RUN_ID" --config "$CFG" \
    || { log "stage 4 failed"; exit 1; }

log "stage 5 (post-hoc: oracle, calibration, length confound, strata)"
uv run python -m halluc.pipeline.stage5_posthoc --run-id "$RUN_ID" --config "$CFG" \
    || { log "stage 5 failed"; exit 1; }

log "pipeline complete"
