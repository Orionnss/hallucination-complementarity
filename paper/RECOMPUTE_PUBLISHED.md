# Recompute: published probes only, no PCA+logreg

Task for an agent with no prior context. Everything needed is below.

**Repo:** `/home/csiouffi/projects/hallucination-complementarity`
**Run with:** `uv run python <script>` from the repo root.

## What to compute

Eight rows, per generator, per dataset, plus pooled:

| row | what it is |
|---|---|
| `saplma` | SAPLMA hidden-state probe |
| `lapeigvals` | LapEigvals, attention Laplacian spectra |
| `icr` | ICR, internal routing |
| `attn_baseline` | attention baseline, raw attention |
| `svd_baseline` | hidden-states baseline, hidden-state spectra |
| `vote_hard` | majority of the five binary decisions |
| `vote_rank` | mean of within-method normalised ranks |
| `vote_soft` | mean of the five probability scores |

Report **both MCC and AUROC** for every cell. Never report MCC alone.

## The one thing that must not go wrong

Two prediction sources exist on disk. They are **not** interchangeable.

| path | what it contains | use it? |
|---|---|---|
| `runs/{run}/stage3_train/pooled/seed{N}/predictions.npz` | each method under its **own published probe** and its own published threshold | **YES** |
| `runs/{run}/stage5_posthoc/block_oof/seed{N}.npz` | every method re-read by a shared **PCA-128 + logistic regression** | **NO** |

Both files contain identically named keys (`preds__saplma`, `scores__saplma`, ...), so
picking the wrong one produces plausible numbers rather than an error. This is the single
most likely way to get this task wrong.

**Verify before computing anything.** Load `main` seed 0 and check SAPLMA's pooled MCC:

* `0.4827` → `stage3_train`, published probe. Correct.
* `0.5482` → `block_oof`, PCA+logreg. Wrong file. Stop and switch.

## Arrays in each file

```python
d = np.load("runs/main/stage3_train/pooled/seed0/predictions.npz", allow_pickle=True)
d["y"]                  # int, 8000, 1 = hallucinated
d["dataset"]            # str, one of triviaqa / nq_open / squad_v2 / coqa
d["groups"]             # str, passage id; CoQA turns and SQuAD questions share passages
d["item_ids"]           # str
d["preds__<method>"]    # int 0/1, the method's published decision
d["scores__<method>"]   # float, the method's probability score
```

Predictions are already out-of-fold under stage 3's nested cross-validation. **Do not
refit any model.** The five methods are consumed exactly as stored.

## Scope

* 6 generators: `main` (Qwen3-14B), `gemma3-12b`, `gemma3-4b`, `llama3.2-3b`,
  `llama3.2-3b-base`, `gemma3-12b-pt`
* 5 seeds each: 0, 1, 2, 3, 4. All 6 generators have all 5.
* 5 scopes: `pooled`, `triviaqa`, `nq_open`, `squad_v2`, `coqa`.
  `pooled` is all 8,000 items at once, **not** the mean of the four dataset columns.
* Report mean and standard deviation across the 5 seeds.

## Vote definitions

Let `P` be the 5×N matrix of published decisions and `S` the 5×N matrix of scores.

```python
votes = P.sum(0)
vote_hard_pred  = (votes >= 3).astype(int)
vote_hard_score = votes / 5.0

soft_score = S.mean(0)

R = np.stack([rankdata(s) / len(s) for s in S])     # scipy.stats.rankdata
rank_score = R.mean(0)
```

`vote_hard` needs no threshold: the five constituent thresholds were already set inside
stage 3's inner splits. `vote_soft` and `vote_rank` need one, and it is the only fitted
parameter anywhere in this task. The scores are averaged with **equal weight** — do not
fit weights.

## Threshold procedure for vote_soft and vote_rank

Fit on the other folds, apply to the held-out fold. Do not fit one threshold and apply it
to everything.

```python
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import matthews_corrcoef

def pick(y, s, idx):
    grid = np.quantile(s[idx], np.linspace(0.05, 0.95, 91))
    return max(grid, key=lambda t: matthews_corrcoef(y[idx], (s[idx] >= t).astype(int)))

pred = np.zeros(len(y), int)
for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(
        s.reshape(-1, 1), y, groups):
    pred[te] = (s[te] >= pick(y, s, tr)).astype(int)
```

Grouping by `groups` is required. CoQA turns share a story and SQuAD questions share a
paragraph, so an ungrouped split puts near-duplicates on both sides.

An earlier version of this code averaged the five per-fold optima and applied that single
threshold to all items. That leaks: each item sits in the training part of four of the
five folds. It inflates pooled MCC by about **+0.0035**. Use the held-out form above.

AUROC uses the raw score and no threshold, so it is unaffected by any of this.

## Reference values to check against

Pooled, 5 seeds, `MCC / AUROC`. Your five single-method rows must match these.

| generator | saplma | lapeigvals | icr | attn_baseline | svd_baseline |
|---|---|---|---|---|---|
| main | 0.4827 / 0.8357 | 0.4682 / 0.8245 | 0.3978 / 0.7779 | 0.4265 / 0.7937 | 0.3322 / 0.7410 |
| gemma3-12b | 0.5314 / 0.8509 | 0.5173 / 0.8444 | 0.4356 / 0.7987 | 0.4704 / 0.8121 | 0.4693 / 0.8168 |
| gemma3-4b | 0.5503 / 0.8464 | 0.4728 / 0.8177 | 0.4600 / 0.8039 | 0.4837 / 0.8145 | 0.4211 / 0.7833 |
| llama3.2-3b | 0.3677 / 0.7529 | 0.2982 / 0.7281 | 0.2102 / 0.6721 | 0.2518 / 0.6847 | 0.1652 / 0.6247 |
| llama3.2-3b-base | 0.4417 / 0.8006 | 0.4604 / 0.8032 | 0.3978 / 0.7743 | 0.4016 / 0.7675 | 0.3438 / 0.7308 |
| gemma3-12b-pt | 0.4298 / 0.7912 | 0.4541 / 0.8167 | 0.3609 / 0.7690 | 0.3770 / 0.7670 | 0.3326 / 0.7355 |

If a single-method row does not match, the wrong source file is being read. Fix that
before looking at the vote rows.

## Output

* Per-seed values written to JSON, not only the aggregates, so paired tests can be run
  later without recomputing.
* A long-format CSV: one row per generator × row × dataset × metric, with mean, sd, and
  the five seed values.
* A table with MCC and AUROC side by side for every cell.

## Out of scope

Do not compute PCA+logreg variants, tuned-MLP variants, PLS variants, CHARM, logprob, or
the learned unions. Only the eight rows listed at the top.
