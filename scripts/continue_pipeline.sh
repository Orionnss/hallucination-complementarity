#!/usr/bin/env bash
# Wait for a running stage 2 to finish, then run stages 3 and 4.
#
# Stage 3 is only started if stage 2 actually produced usable labels for every dataset —
# training on a truncated or empty label set would silently produce meaningless numbers
# rather than failing.
#
# Usage: scripts/continue_pipeline.sh <stage2_pid> [n_jobs]

set -uo pipefail

STAGE2_PID="${1:?usage: continue_pipeline.sh <stage2_pid> [n_jobs]}"
N_JOBS="${2:-32}"
CFG=/dev/null

echo "[$(date '+%F %T')] waiting for stage 2 (pid $STAGE2_PID)"
while kill -0 "$STAGE2_PID" 2>/dev/null; do sleep 60; done
echo "[$(date '+%F %T')] stage 2 process exited"

# Guard: every dataset must have labels with a non-degenerate class balance.
if ! uv run python - <<'PY'
import sys, json
from pathlib import Path
ok = True
for ds in ["triviaqa", "nq_open", "squad_v2", "coqa"]:
    p = Path(f"runs/main/stage2_judge/{ds}/labels.json")
    if not p.exists():
        print(f"  {ds}: MISSING labels.json"); ok = False; continue
    d = json.load(open(p))
    rate = d.get("hallucination_rate_scored")
    print(f"  {ds}: labelled={d['n_labelled']} scored={d['n_scored']} "
          f"dropped={d['n_dropped_invalid']} halluc_rate={rate} "
          f"unanimous={d['agreement']['unanimous_rate']}")
    if d["n_scored"] < 500 or rate is None or not (0.02 < rate < 0.98):
        print(f"  {ds}: UNUSABLE (too few scored items, or degenerate class balance)")
        ok = False
sys.exit(0 if ok else 1)
PY
then
    echo "[$(date '+%F %T')] stage 2 output failed validation; not starting stage 3"
    exit 1
fi

echo "[$(date '+%F %T')] starting stage 3 (pooled training, per-dataset eval, n_jobs=$N_JOBS)"
uv run python -m halluc.pipeline.stage3_train --n-jobs "$N_JOBS" --config "$CFG"
rc=$?
if [ $rc -ne 0 ]; then
    echo "[$(date '+%F %T')] stage 3 failed (rc=$rc); not starting stage 4"
    exit $rc
fi

echo "[$(date '+%F %T')] starting stage 4 (complementarity analysis)"
uv run python -m halluc.pipeline.stage4_analysis --config "$CFG"
echo "[$(date '+%F %T')] pipeline complete"
