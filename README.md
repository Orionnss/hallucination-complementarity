# Hallucination detector ensembling — published-probe pipeline

Code to reproduce one line of work: generate and label QA answers, train each hallucination
detector **under its own published configuration**, and then study how those detectors
combine — voting ensembles, pairwise agreement, and how performance moves as voters are
added or removed.

This branch is a pruned copy of the full research repository. Everything not needed for
that line is removed; see [What is not here](#what-is-not-here).

## The rule that governs everything

Every detector is used exactly as its own paper specifies: its published probe **and** its
published decision threshold. Nothing is refitted at the analysis stage, and no shared
classifier is substituted for a detector's own.

The repository can produce two prediction sets that look alike and are not:

| path | contents |
|---|---|
| `runs/{run}/stage3_train/pooled/seed{N}/predictions.npz` | each detector under its **own published probe** — what this branch uses |
| `runs/{run}/stage5_posthoc/block_oof/seed{N}.npz` | every feature block re-read by a shared PCA-128 + logistic regression |

Both files use identical key names (`preds__saplma`, `scores__saplma`, …), so reading the
wrong one yields plausible numbers rather than an error. Every analysis script on this
branch reads the first. To verify after a run, check SAPLMA's pooled MCC on `main`:
**0.4827** is the published probe; **0.5482** means the PCA+logreg file was read.

## Install

```bash
uv sync
```

Python 3.12. GPU needed for stages 1–2 only; everything after is CPU-bound scikit-learn.

## Pipeline

Stages are separate entry points so they never share a process, and each resumes from disk.

```bash
# 1. generate answers and extract features (GPU, ~24 h per generator)
uv run python -m halluc.pipeline.stage1_extract --run-id main --model Qwen/Qwen3-14B --device cuda:0

# 2. label with the 3-judge pool (GPU, 4-bit, ~2 h)
uv run python -m halluc.pipeline.stage2_judge --run-id main --device cuda:0

# 3. train every detector under its published configuration (CPU)
uv run python -m halluc.pipeline.stage3_train --run-id main --scope pooled

# 4. per-dataset metrics, kappa and McNemar between detectors
uv run python -m halluc.pipeline.stage4_analysis --run-id main --no-charm
```

`scripts/run_downstream.sh` chains stages 1–5 for one generator;
`scripts/continue_pipeline.sh` resumes from stage 2 when extraction is already banked.

Stage 3 writes `predictions.npz` per seed, holding each detector's out-of-fold decisions
and scores. Everything below reads those files and refits nothing.

### Protocol that makes the numbers comparable

- **Grouped folds.** CoQA turns share a story and SQuAD questions share a paragraph, so
  folds are grouped by passage. Random folds would put near-duplicates on both sides.
- **Stratified by (dataset, label).** Hallucination rates range 0.15–0.66 across slices.
- **Nothing fit on test.** Scalers, probe weights, hyperparameters and thresholds all come
  from training folds; selection uses an inner split.
- **5 seeds × 5 outer folds** per generator. 8,000 items per seed, 2,000 per dataset.

`DESIGN.md` documents the protocol in full.

## Analysis

All scripts take `--runs`/`--run`, write JSON plus a long-format CSV under `runs/`, and
report both MCC and AUROC. Per-seed values are stored, not just aggregates, so paired tests
need no recomputation.

### Ensembling voters

```bash
uv run python scripts/vote_recompute.py --source published
```

Hard, rank and soft voting over the five detectors, per dataset and per generator. Hard
voting counts the published decisions and estimates nothing; soft and rank voting average
scores or within-method ranks and fit one threshold by grouped 5-fold cross-fitting, so no
item is scored under a threshold fitted on its own label.

`--source pcalr` exists for comparison with the PCA+logreg reader. It is **not** part of
this line of work.

### Pairwise correlations

```bash
uv run python scripts/pairwise_correlations.py            # between detectors and votes
uv run python scripts/conditional_correlations.py --perms 30   # on each other's error sets
```

The first reports Cohen's kappa, raw agreement, Pearson and Spearman for all 28 pairs.
Kappa and agreement are computed on the decisions, Pearson and Spearman on the scores;
forcing all four onto one representation would make them incomparable. Vote–detector pairs
are partly definitional and are marked.

The second restricts to the items one detector got wrong and measures another against the
label there. Those values cannot be read against zero — conditioning on a detector's errors
is collider conditioning, and the slice's class balance shifts on its own — so each cell
carries a null built by permuting the evaluated detector within each class. `--perms`
controls the permutation count.

### Number of voters

```bash
uv run python scripts/voter_ablation.py      # all 31 subsets of the five detectors
uv run python scripts/voter_ablation3.py     # SAPLMA / LapEigvals / ICR, published thresholds
uv run python scripts/voter_marginal.py      # reads voter_ablation.json
```

`voter_ablation` sweeps every subset under all three rules. At k=1 it re-thresholds the
single detector by the combiners' rule, which keeps the sweep internally consistent but
means those cells are **not** the detector's published figure.

`voter_ablation3` avoids that: it restricts to three detectors and keeps each one's
published threshold at k=1, so the k=1 row is the reported number and the k=1→k=2 step
isolates the effect of adding a voter from the effect of re-thresholding.

`voter_marginal` reports, per voter, the cost of removing it from the full ensemble and its
exact Shapley value over all 16 coalitions that exclude it. The two disagree when a voter
is redundant with another, which is the point of reporting both.

## What is not here

Removed from this branch, available on `main`:

- CHARM (stage 6 and `src/halluc/charm/`)
- the shared PCA+logreg reader sweeps, tuned-MLP variants, PCA width sweeps
- PLS and supervised-reduction work, component importance, layer ablations, LayerMix
- mixture-of-dataset-experts, cascades, two-stage correctors, rescue analyses
- cross-dataset transfer, risk–coverage, label audit, learning curves

Two things could not be removed without editing source, which was out of scope for this
branch:

- `build_detectors` in `src/halluc/detectors.py` still registers `union_raw` and
  `union_equal`, so stage 3 trains them and writes their columns. No analysis script on
  this branch reads them.
- `src/halluc/config.py` keeps a `CharmConfig` dataclass and `stage4_analysis` still looks
  for a stage-6 predictions file. Both are inert when stage 6 has not run; pass
  `--no-charm` to stage 4 to skip the lookup entirely.

## Layout

```
src/halluc/
  pipeline/      stages 1–5, each a standalone entry point
  features/      SAPLMA, LapEigvals, ICR, attention and hidden-state baselines
  detectors.py   each detector's published probe and hyperparameter grid
  judges/        the 3-model judge pool
  datasets/      TriviaQA, NQ-Open, SQuAD v2, CoQA loaders
scripts/         the analysis above, plus two pipeline runners
runs/            all outputs (gitignored)
```
