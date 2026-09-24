# Revisiting Hallucination Probes for Complementary Detection

Code for the paper. This branch contains only what is needed to produce its tables:
generation and labelling, the published detector probes, the three voting rules, and the
correlation and ablation analyses. Everything else from the research repository is on
`main`.

## The rule that governs every number

Each detector is used exactly as its own paper specifies: its published probe **and** its
published decision threshold. Nothing is refitted at the analysis stage.

Two prediction files exist and look alike:

| path | contents |
|---|---|
| `runs/{run}/stage3_train/pooled/seed{N}/predictions.npz` | each detector under its **own published probe**, used everywhere here |
| `runs/{run}/stage5_posthoc/block_oof/seed{N}.npz` | every block re-read by a shared PCA-128 + logistic regression, **not** used |

They share key names (`preds__saplma`, `scores__saplma`, ...), so reading the wrong one
gives plausible numbers rather than an error. Check before trusting any output: SAPLMA's
pooled MCC on `main` is **0.4827** under the published probe and **0.5482** under PCA+logreg.

## Install

```bash
uv sync
```

Python 3.12. GPU required for stages 1 and 2 only; every analysis below is CPU-bound
scikit-learn.

## Pipeline (run once per generator)

```bash
uv run python -m halluc.pipeline.stage1_extract --run-id main --model Qwen/Qwen3-14B --device cuda:0
uv run python -m halluc.pipeline.stage2_judge   --run-id main --device cuda:0
uv run python -m halluc.pipeline.stage3_train   --run-id main --scope pooled
```

Generators used in the paper: `main` (Qwen3-14B), `gemma3-12b`, `gemma3-4b`, `llama3.2-3b`,
`llama3.2-3b-base`, `gemma3-12b-pt`. `scripts/run_downstream.sh` chains the stages for one
generator; `scripts/continue_pipeline.sh` resumes from stage 2.

Stage 3 writes, per seed, the out-of-fold decisions and scores of every detector. All five
analyses below read those files and refit nothing. Folds are grouped by passage (CoQA turns
share a story, SQuAD questions share a paragraph) and stratified by dataset and label;
5 seeds, 5 outer folds, 8,000 items per seed. `DESIGN.md` documents the protocol.

## Analyses

Four commands produce every number in the paper. Each writes a JSON with per-seed values and
a long-format CSV under `runs/`.

```bash
uv run python scripts/vote_recompute.py --source published   # voting rules vs individual detectors
uv run python scripts/pairwise_correlations.py               # agreement between detectors and votes
uv run python scripts/conditional_correlations.py --perms 30 # agreement on another detector's errors
uv run python scripts/voter_ablation.py                      # all 31 subsets, three rules
uv run python scripts/voter_marginal.py                      # removal cost and Shapley value
```

`voter_marginal.py` reads `runs/voter_ablation.json`, so run it after `voter_ablation.py`.
`conditional_correlations.py` is the slowest at roughly 20 minutes; the others take a few
minutes each once stage 3 has run.

## Which command produces which table

| Table | Content | Produced by | Data file |
|---|---|---|---|
| 1 | Voting combiners vs individual detectors, per dataset and generator | `vote_recompute.py` | `runs/vote_recompute.json` |
| 2 | Average pairwise Cohen's kappa between voters | `pairwise_correlations.py` | `runs/pairwise_correlations.json` |
| 3 | Average pairwise Pearson r between voters | `pairwise_correlations.py` | `runs/pairwise_correlations.json` |
| 4 | Kappa on another detector's errors, with the independence null | `conditional_correlations.py` | `runs/conditional_correlations.json` |
| 5 | Observed minus null, averaged over the 20 ordered pairs | `conditional_correlations.py` | `runs/conditional_correlations.json` |
| 6 | AUROC and MCC at 1 vs 4 voters | not produced by any script here | |
| 7 | What predicts a combination's score and its gain | `voter_ablation.py` + `pairwise_correlations.py` | `runs/voter_ablation.json` |
| 8 | Spread illustrated on Qwen3-14B | `voter_ablation.py` | `runs/voter_ablation.json` |
| 9-11 | Four metrics, means per generator, means per dataset | `pairwise_correlations.py` | `runs/pairwise_correlations.json` |
| 12-14 | Duplicates of Tables 9-11 | `pairwise_correlations.py` | `runs/pairwise_correlations.json` |
| 15 | The three vote rules against each other | `pairwise_correlations.py` | `runs/pairwise_correlations.json` |
| 16-17 | Removal cost and Shapley value, overall and per model | `voter_marginal.py` | `runs/voter_marginal.json` |
| 18 | All 31 subsets, averaged over generators | `voter_ablation.py` | `runs/voter_ablation.json` |
| 19 | Shapley value per generator | `voter_marginal.py` | `runs/voter_marginal.json` |
| 20-21 | Predictors of gain, and of absolute score | `voter_ablation.py` + `pairwise_correlations.py` | `runs/voter_ablation.json` |
| 22 | Best-member quality vs absolute score, per generator | `voter_ablation.py` | `runs/voter_ablation.json` |
| 23 | Worked example on Qwen3-14B | `voter_ablation.py` | `runs/voter_ablation.json` |

The committed `paper/*.tex` files are the tables as typeset. Each was produced from the JSON
above; the formatting scripts are not on this branch, so regenerating a table means rerunning
its analysis and reading the numbers from the JSON or CSV.

| `paper/*.tex` | Tables |
|---|---|
| `vote_results.tex` | 1 |
| `pairwise_corr.tex` | 2, 3 |
| `conditional_corr.tex` | 4, 5 |
| `quality_gain_main.tex` | 7, 8 |
| `appendix_correlation.tex` | 9-17 |
| `appendix_voters.tex` | 18, 19 |
| `appendix_quality_gain.tex` | 20-23 |
| `alg_vote.tex` | Algorithms 1-2 and the instantiation table |
