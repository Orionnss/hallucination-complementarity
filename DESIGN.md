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

Five method entries plus two union comparators, all reading features produced in a
**single extraction loop** (stage 1), so the generator runs exactly once per pool item —
plus CHARM, which cannot use that loop's output and owns a second one (see below).

| Name | Features (Qwen3-14B: L=40, H=40, d=5120) | Dim | Head |
|---|---|---|---|
| `lapeigvals` | top-k(10) of `sort(diag(D−A))` per (layer,head); `d_ii=(Σ_u a_ui)/(T−i)` | 16,000 → PCA | logistic reg. |
| `attn_baseline` | top-k(10) of `sort(diag(A))` per (layer,head) — raw attention, **no Laplacian, no PCA** | 16,000 | logistic reg. |
| `svd_baseline` | singular values of last-token hidden states stacked over all layers `[41, 5120]` | 41 | logistic reg. |
| `saplma` | last-token hidden state, one selected layer | 5,120 | MLP (256,128,64) |
| `icr` | `JSD(softmax(Proj^ℓ_i), Attn^ℓ_i)` pooled over layers | ~40 | MLP (4 layers) |
| `union_raw` | z-scored concat of all blocks | ~37k | logistic reg. |
| `union_equal` | each block PCA'd to 128 first, then concat | 640 | logistic reg. |
| `charm` | whole attention graph: `X_E=α_ij` per edge, `X_V=(α_ii ‖ a_i^l)` per token | ragged | GNN (msg-passing) |

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

### CHARM: trained during extraction, never persisted

CHARM (arXiv 2509.24770, ICLR 2026) is the one method that does not fit the shard
scheme at all. Its input is not a fixed-width vector per sample but a whole attributed
graph: tokens are nodes, an edge (i,j) means token i attends to token j, each edge
carries the `L·H`-dimensional vector of that pair's attention across every layer and
head, and each node carries its self-attention concatenated with its residual-stream
activations. A dense `X_E` runs to tens of MB per item at Qwen3-14B's `L·H = 1,600`, and
the shape is ragged in the sequence length, so there is nothing to stack into an `.npz`.

So `stage6_charm` runs **its own extraction loop and trains from a RAM cache**. Two
properties keep this honest:

- **The protocol is stage 3's, unchanged.** Same `draw_and_combine` per-seed draw, same
  balanced per-dataset sampling, same grouped folds stratified on (dataset, label), same
  inner split for hyperparameters and threshold, same 5 seeds × 5 folds. Only the
  feature pipeline differs, so CHARM-vs-rest isolates the method and not the protocol.
- **The generator is not re-run.** Stage 1 recorded what the model said under greedy
  decoding, so this stage re-encodes that answer and does one forward pass instead of up
  to 256 decode steps. Items whose text does not re-encode to the same token count
  (~0.1%, measured) fall back to a real `generate()`, so the trace always corresponds to
  a sequence the model actually produced.

Two deviations, both narrowing what CHARM sees rather than widening it:

- **No refit on the full training fold.** A network needs a held-out split to early-stop
  on, so the reported model trains on inner-train and stops on inner-val — the same
  split its hyperparameters and threshold come from. It sees *less* data than the probes
  it is compared against, never more.
- **A pinned hyperparameter grid.** Table 9's published space is 576 points, which at 5
  seeds × 5 folds would be 14,400 network fits. The default grid varies capacity and
  depth and pins the rest at the paper's values, the same way `lapeigvals` is pinned
  after a prior sweep. `--grid full` restores the published search space.

Making this tractable rests on one observation. `X_E^τ` is sparsified at τ=0.05
(Equation 1), and because an attention row sums to 1, at most 20 of a row's entries can
survive — measured density on Qwen3-14B is **2–5% of a dense `X_E`**. The graph is
therefore held in a CSR layout over (edge, channel) non-zeros, which is also exactly
what `nn.EmbeddingBag` consumes, so the model applies `msg`'s edge weights to the sparse
features directly and never densifies them either.

Measured on Qwen3-14B (L·H=1,600, d=5,120, one activation layer at 0.7 depth), that puts
the whole four-dataset cache at **~50 GiB of RAM** for ~15,100 scored items, and the
extraction pass at **~25 minutes**:

| Dataset | MB/graph | dense `X_E` alone | mean edges | density | projected |
|---|---|---|---|---|---|
| TriviaQA | 2.01 | 7.6 MB | 2,489 | 4.4% | 7.8 GiB |
| NQ-Open | 2.04 | 7.7 MB | 2,516 | 4.6% | 7.1 GiB |
| SQuAD v2 | 1.65 | 5.2 MB | 1,696 | 4.9% | 5.8 GiB |
| CoQA | 7.57 | 7.7 MB | 2,508 | 1.9% | 29.1 GiB |

CoQA dominates through its *node* features, not its edges: a 543-token passage carries
543 × (1,600 + 5,120) half-precision values whatever the attention does. `--tau` trades
edge fidelity for footprint and `--act-fractions` trades node fidelity; the run aborts
rather than swaps if the cache would exceed `--max-cache-gib`.

Following the paper's own experimental form, prompt→prompt edges are dropped, which is
what bounds the edge count on CoQA's long passages; pooling is over response tokens
only, since prompt tokens then have no incoming messages.

**Resume.** Training decomposes into (seed, fold, grid point) fits — 100 at the default
grid, several hours — and each is checkpointed as it finishes, with its scores, its
record, and its early-stopped weights. A restart replays the completed ones and trains
only what is missing. A seed that already has `metrics.json` is skipped outright, and
when that accounts for every seed the generator is never loaded at all (~6 s, no GPU).

Three guards stop a resume from blending results computed over different data:

- the **seed skip is fingerprint-checked**, not existence-checked. `metrics.json`
  records a hash of that seed's drawn item ids, and a run whose draw does not match
  re-runs the seed instead of accepting it. This is what stops, say, a `--datasets
  triviaqa` probe from satisfying the skip for a four-dataset run — the per-dataset
  draws coincide, but the seed's sample is 2,000 items rather than 8,000, and the cached
  score arrays are positional.
- the **fit store carries the same fingerprint**, so a checkpoint directory belonging to
  a different draw is refused rather than half-reused;
- each fit stores **its own hyperparameters** and is keyed by their hash, so an edited or
  reordered grid re-trains the affected points instead of loading someone else's scores
  under a reused index.

The graph cache is deliberately *not* checkpointed — that is the whole design — so a
resumed run still pays the ~25-minute extraction pass. Only the expensive half is
recovered.

Implemented in plain PyTorch (`index_add_` + `EmbeddingBag`) rather than PyTorch
Geometric — the single message-passing form is a few lines of scatter, and the
dependency is not worth pinning against this project's torch build.

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
  stage6_charm/<scope>/seed<k>/
    predictions.npz               CHARM out-of-fold probabilities + decisions
    metrics.json                  AUROC/MCC/threshold per fold
    checkpoints/
      manifest.json               fingerprint of this seed's sample draw
      fit_f<fold>_<params>.npz    one completed fit: val + test scores, record
      fit_f<fold>_<params>.pt     that fit's early-stopped weights
                                  (the graphs themselves are never written)
  stage4_analysis/
    kappa.json  mcnemar.json  summary.json
```

## Known caveats to report with results

- **`lapeigvals` vs `attn_baseline` confounds the Laplacian with PCA.** LapEigvals
  applies PCA because that is part of the published method; the baseline is specified as
  no-PCA. A gap between them therefore reflects "Laplacian + PCA" versus "raw attention
  at 16,000 dims", and cannot be attributed to the Laplacian on its own. Report the
  comparison in those terms rather than as evidence about the Laplacian.
- **CHARM is not compared on a like-for-like fit budget.** It is the only learned,
  end-to-end detector here; the probes it is measured against are logistic regressions
  and small MLPs over fixed features. A gap in either direction reflects the whole
  method — graph structure, message passing, and capacity together — not the graph
  representation on its own. The paper's own `CHARM (no g.)` ablation is the control
  that separates those, and it is not run here.
- **CHARM's scores are reproducible to ~1e-4, not bitwise.** Message passing aggregates
  with `index_add_` and `EmbeddingBag`, whose backward passes sum gradients in thread
  (or CUDA atomic) completion order. That is two orders of magnitude below the
  across-seed spread reported, but it means a rerun is not byte-identical unless
  `TrainConfig.deterministic` is set. Every other detector here is deterministic. The
  same applies across a crash: fits replayed from checkpoint are exact, but any fit that
  has to be *re-trained* lands ~1e-4 from where the uninterrupted run would have put it,
  so a resumed run is not byte-identical to a clean one.
- **The CHARM graph cache is RAM-only and not resumable.** That is the point — nothing
  is written to disk — so a crash always re-pays the ~25-minute extraction pass, even
  though the training fits themselves resume from checkpoint.
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
- CHARM's activation layer is fixed at 0.7 of model depth rather than tuned, unlike
  SAPLMA's. The paper reports (Table 7) that CHARM is robust to this choice and that
  concatenating several layers helps slightly; `--act-fractions` takes any number of
  them, and `--act-fractions` with no values gives the attention-only `CHARM (att)`.
- CHARM's response span starts at `prompt_len - 1`, matching how `features/icr.py`
  treats the position that commits to the first answer token. The paper's `n_p`/`n_r`
  split is literal, so this includes one more token than a strict reading — the same
  token every last-token probe in this repo reads.
