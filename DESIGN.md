# Hallucination Detection Complementarity — Design

Study question: do existing hallucination-detection methods pick up on **the same**
hallucinations, or **complementary** ones? Answered with per-method MCC against a
judge-labelled reference, plus pairwise Cohen's κ and McNemar tests between predictors.

## Locked decisions

| Decision | Value |
|---|---|
| Generator | `Qwen/Qwen3-14B`, **bf16** (unquantized — probes read these activations) |
| Generation | greedy, `enable_thinking=False`, `max_new_tokens=256` |
| Judges | 3 local models, **4-bit NF4**, majority vote |
| Staging | generator and judges **never co-resident** — stage 1 unloads before stage 2 |
| Pool / N / seeds | 4000 per dataset (NQ-Open capped at 3610), N=2000 per seed, 5 seeds |
| Seed semantics | seed selects **which samples** are drawn from the pool (+ CV splits + probe init) |
| Labels | `HALLUCINATED` / `NOT_HALLUCINATED` / `INVALID`; INVALID dropped from train **and** eval |
| Metrics | AUROC primary (threshold-free); MCC at a threshold tuned on a validation split |
| Agreement | pairwise Cohen's κ + McNemar between every predictor pair, and judge-vs-judge |

### Datasets — context policy

Closed-book unless the task is undefined without a passage (your rule; also what the
LapEigvals paper does):

| Dataset | Split | Size | Context |
|---|---|---|---|
| TriviaQA (`rc.nocontext`) | validation | 7,983 | **none** |
| NQ-Open | validation | 3,610 | **none** |
| SQuAD v2 (`rc.nocontext`) | validation | 9,960 | **none** |
| CoQA | dev | 5,928 | **passage kept** — only dataset that needs it |

### Judge pool

`google/gemma-3-12b-it` (base) + `Qwen/Qwen2.5-14B-Instruct` +
`mistralai/Mistral-Nemo-Instruct-2407`. Three distinct model families so agreement is
meaningful rather than shared-pretraining artifact. All already in the local HF cache.
Judges see question, gold answer(s), and the model's answer. Label = majority vote;
ties → `INVALID`. **Denial of the premise counts as HALLUCINATED.** Per-pair agreement
rate and Cohen's κ are written into the stage-2 JSON.

## Detectors

Five method entries plus two union comparators. All read features produced in a
**single extraction loop** (stage 1), so the generator runs exactly once per pool item.

| Name | Features (Qwen3-14B: L=40, H=40, d=5120) | Dim | Head |
|---|---|---|---|
| `lapeigvals` | top-k(10) of `sort(diag(D−A))` per (layer,head); `d_ii=(Σ_u a_ui)/(T−i)` | 16,000 → PCA | logistic reg. |
| `attn_baseline` | top-k(10) of `sort(diag(A))` per (layer,head) — raw attention, **no Laplacian, no PCA** | 16,000 | logistic reg. |
| `svd_baseline` | singular values of last-token hidden states stacked over all layers `[41, 5120]` | 41 | logistic reg. |
| `saplma` | last-token hidden state, one selected layer | 5,120 | MLP (256,128,64) |
| `icr` | `JSD(softmax(Proj^ℓ_i), Attn^ℓ_i)` pooled over layers | ~40 | MLP (4 layers) |
| `union_raw` | z-scored concat of all blocks | ~37k | logistic reg. |
| `union_equal` | each block PCA'd to 128 first, then concat | 640 | logistic reg. |

`attn_baseline` is deliberately the exact shape of `lapeigvals` with the Laplacian
removed, so the pair isolates *"does the Laplacian transform add anything over raw
attention?"* — otherwise the two would be confounded.

Both union variants are reported: `union_raw` shows whether concatenation helps at all,
`union_equal` shows whether it helps once LapEigvals' 16k-dim advantage is neutralized.

### Extraction mechanics

`generate()` returns per-step attentions that are awkward to assemble, so stage 1 does:
generate greedily → **re-forward the full prompt+answer sequence once** with
`output_attentions=True, output_hidden_states=True` (eager attention) → compute every
method's features from that one pass. `T` covers prompt + generation, per LapEigvals.

ICR needs `Δx^ℓ_i` for *all* token positions — storing those would be ~126 MB/sample, so
ICR is reduced to its ~40 scores inside the loop and only those are persisted. This is
exactly why features are computed during extraction rather than from dumped activations.

## Evaluation

**Training is pooled across datasets by default; evaluation is per dataset.** One probe
is fitted on the union of all four datasets and then scored separately on each, which
asks "does a single detector work everywhere?" rather than "does a TriviaQA-fitted
detector work on TriviaQA". `--scope per_dataset` restores independent training.

Pooling adds two requirements:

- **Balanced draw.** `n_per_seed` items are drawn from *each* dataset's pool, not from
  the pooled set, so the largest dataset cannot dominate the shared probe and reduce the
  smaller datasets to a pure transfer test.
- **Folds stratified by (dataset, label).** The datasets have different hallucination
  rates; stratifying on the label alone would let a fold drift toward one dataset and
  leave its per-dataset metrics resting on a handful of rows. Group ids are
  dataset-prefixed, so they remain unique after pooling.

The probe is global, so its threshold is global too — tuned on the inner validation
split. Per-dataset thresholds are recorded alongside, to quantify what a single
operating point costs on each dataset.

Nested cross-validation, repeated across 5 seeds:

- outer 5-fold → test set, never touched by fitting
- inner split → PCA fit, probe hyperparameters, and the MCC-maximizing threshold
- PCA and standardizer are fit on **training folds only** (leakage guard)
- per (dataset, seed, fold, method): AUROC, MCC, accuracy, and the raw per-sample
  predictions — the predictions are what κ and McNemar are computed from
- κ / McNemar computed on pooled out-of-fold predictions, per dataset, per seed, then
  aggregated across seeds with mean ± std

## Persistence

Every stage writes JSON metadata and resumes from checkpoints. Float features go in
`.npz` sidecars referenced by the JSON — 16k floats per sample is not JSON-shaped.

```
runs/<run_id>/
  config.json                     resolved config + git sha + library versions
  stage1_extract/<ds>/
    manifest.json                 per-item metadata, generation, timings, feature refs
    features_<shard>.npz          float arrays
    checkpoint.jsonl              append-only completed ids (resume)
  stage2_judge/<ds>/
    labels.json                   per-item votes, final label, per-judge rationale
    agreement.json                pairwise agreement rate + Cohen's κ per judge pair
  stage3_train/<ds>/seed<k>/
    folds.json                    split indices
    predictions.npz               per-method out-of-fold probabilities + hard decisions
    metrics.json                  AUROC/MCC/threshold per method per fold
  stage4_analysis/
    kappa.json  mcnemar.json  summary.json
```

## Known caveats to report with results

- **`lapeigvals` vs `attn_baseline` confounds the Laplacian with PCA.** LapEigvals
  applies PCA because that is part of the published method; the baseline is specified as
  no-PCA. A gap between them therefore reflects "Laplacian + PCA" versus "raw attention
  at 16,000 dims", and cannot be attributed to the Laplacian on its own. Report the
  comparison in those terms rather than as evidence about the Laplacian.
- **TriviaQA counts differ from the LapEigvals paper.** They report 7,983 validation
  items (consistent with `unfiltered.nocontext`); `rc.nocontext` deduplicated by
  question_id gives 9,960. Preprocessing was not reverse-engineered.

All performance claims must come from the real pipeline. Synthetic-data runs are used
only to check that the CV machinery executes, that grouped folds hold, and that no test
data leaks — never to say anything about how the methods compare.

## Open items

- ICR layer-pooling is described as "pooled ICR Score" in §4.2 of arXiv 2507.16488;
  exact pooling operator is in the appendix and not yet confirmed. Implemented as
  attention-weighted pooling over layers, flagged in code for verification.
- SAPLMA's probe layer is a hyperparameter; selected on the inner fold rather than fixed.
