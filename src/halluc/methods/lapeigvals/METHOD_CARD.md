# LapEigvals

## Source
- Paper: Binkowski et al., *Hallucination Detection in LLMs Using Spectral Features of
  Attention Maps*, EMNLP 2025, arXiv 2502.17598v2
- Code: https://github.com/graphml-lab-pwr/lapeigvals @ `74f885c`, **no licence** (see
  SOURCE.md)
- Status: **adapted**. The adapter calls the official functions from an unchanged
  checkout.

## Type
A — supervised probe (ADDING_A_METHOD.md §3).

## Pipeline

Upstream paths are relative to `original-repos/lapeigvals`.

| Part | Paper | Code | Our adapter |
|---|---|---|---|
| Input | Attention maps of all tokens, prompt and answer (§3). | `model.generate(output_attentions=True)`, eager attention, bf16, batch size 1: `hallucinations/llm/predict.py:15`, `config/llm/*.yaml:6-7`, `dvc.yaml:38` | **Harness, same call.** The shared trace *is* this `generate()` output (eager, bf16, batch size 1, greedy), cut to the kept answer by the harness. No conversion (since 2026-10-06; before, a re-forward, see D1). |
| Preprocessing | — | `stack_attention_matrix` (`hallucinations/features/attention_weights.py:156`): per-step rows → one [H, T′, T′] per layer, where T′ = generated length − 1. `remove_padding_from_intermediate_states` (`hallucinations/features/processing.py:8`): removes pad tokens at both ends and checks that rows sum to 1. | The adapter calls `stack_attention_matrix` on the native view (moved to CPU, as upstream does) and `remove_padding_from_intermediate_states` with the trace's ids, without change. |
| Feature computation | L = D − A (Eq. 1). d_ii = Σ_u a_ui / (T − i) (Eq. 2). The eigenvalues are the diagonal of L, because L is lower-triangular (Eq. 3). Top-k per head, all layers. | `laplacian_diagonal_from_attn(..., vertical_edges=False)` (`attention_weights.py:24`), on CPU, in the attention dtype (bf16). Cast to float32 after (`scripts/probes/train_attn_vs_laplacian.py:89-90`). `get_laplacian_eigvals_per_head_topk(layer_idx=None)` (`hallucinations/features/attn_feats.py:49`). | Calls these three functions without change. Stores top-100 per head; each smaller k is a prefix. |
| Reader | PCA 512, then logistic regression (scikit-learn, `max_iter=2000`, `class_weight="balanced"`). Standardisation is not mentioned. | `train_logistic_regression` (`hallucinations/probe_models/lr.py:19`): `Pipeline([PCA, LogisticRegression])`, **no scaler**, C = 1.0 (default), PCA width = min(512, n_features, n_rows) (`hallucinations/probe_models/commons.py`), `--pca-dim 512` (`dvc.yaml:189`). | `reader.OfficialProbe` calls `train_logistic_regression` without change. Reader name: `lapeigvals_official`. |
| Decision | Threshold 0.5 for precision and recall (App. G.1). | `test_proba.argmax(axis=-1)` (`lr.py:86`) = p > 0.5. | Recorded as `native_preds__…`. The main results use the harness threshold (MCC on the inner split). |
| k selection | k ∈ {5, 10, 20, 50, 100}. "Selected the result with the highest efficacy" (§4.2). The paper does not name a validation split. | `TOP_K_EIGVALS = [5, 10, 25, 50, 100]` (`train_attn_vs_laplacian.py:23`), filtered to k ≤ shortest item (line 40). Each k is trained and saved. The code does not select a k. | k is a view hyperparameter. The k filter is the official one. The harness selects k on the inner split by AUROC. |
| Split | 80/20, stratified on the label (App. E). | `train_test_split(test_size=0.2, stratify=labels, random_state=42)` (`scripts/dataset/generate_split.py`, `dvc.yaml:171`). One split, no groups, no validation split. | **Harness.** Grouped nested CV, 5 seeds × 5 folds. |
| Labels | gpt-4o-mini as judge (§4.1). | `scripts/eval/llm_as_judge.py` | **Harness.** Majority of three local judges. |

## Feature blocks

| Block | Type | Shape per item | dtype | Content |
|---|---|---|---|---|
| `lapeigvals_official` | vector (via the k view) | [L, H, 100] | float32 | Official top-k Laplacian eigenvalues, sorted in descending order. NaN after T″. |
| `attneigvals_official` | vector (via the k view) | [L, H, 100] | float32 | Official top-k attention eigenvalues (the paper's AttnEigvals control). |
| `lapeigvals_official_T` | scalar | [1] | int32 | T″, the number of tokens after the official preprocessing. The k filter uses it. |

Store: `runs/<run>/methods/lapeigvals_official/<dataset>/`.

## Native reader
- Hyperparameters searched: none in the code. The k of the feature view is selected on
  the inner split (see deviation D3).
- Hyperparameters fixed: PCA 512, C = 1.0, `class_weight="balanced"`,
  `max_iter=2000`, no standardisation.
- Number of configurations: up to 5 (one for each k that the k filter keeps).

## Parts of the source code not used

| Part | Why |
|---|---|
| `hallucinations/data/*`, `config/dataset/*`, `config/prompt/*` | The harness gives the items and the generation prompt (rule R1). |
| `hallucinations/llm/predict.py`, `activation_storage.py`, `feature_storage.py` | The generation loop and the storage. The harness owns generation (R2). The adapter rebuilds the same input (test F1). |
| `scripts/eval/*`, `scripts/dataset/generate_labels*.py` | Labelling. The harness owns the labels. |
| `scripts/dataset/generate_split.py` | Split. The harness owns the splits. |
| `hallucinations/features/laplacian.py` | A second Laplacian with edges between layers. The published pipeline does not call it. |
| `hallucinations/__init__.py` | It sets `torch.set_float32_matmul_precision("high")` for the whole process. The adapter does not execute it. |
| AttnLogDet, `probe_attn_score.py`, hidden-state baselines | Other baselines in the same repository. They are not part of LapEigvals. |

## Paper versus code differences

| Item | Paper | Code | Default we use |
|---|---|---|---|
| k values | {5, 10, 20, 50, 100} | {5, 10, 25, 50, 100} | Code |
| Standardisation before PCA | Not mentioned | None | Code (none) |
| C of logistic regression | Not mentioned | 1.0 (scikit-learn default) | Code |

## Patches

| Id | What | Why | Effect on the computation |
|---|---|---|---|
| P1 | In the upstream `lr` module only, `roc_auc_score` and `average_precision_score` return `np.float64`. | Dependency version. Upstream calls `.item()` on the result. scikit-learn 1.6.1 returns `np.float64`, and 1.9.0 returns a Python float. | None. These calls compute train and test metrics, which the adapter discards. The file on disk does not change. |

## Deviations

| Id | What | Why | Direction | Test |
|---|---|---|---|---|
| D1 | **Removed on 2026-10-06.** Until then the attention came from one full forward pass, not from incremental decoding with a KV cache: in bf16 the largest top-10 difference per item was about 0.03. The shared trace now gives the adapter the `generate()` output itself. | — | — | — |
| D2 | Generation prompt, generators, datasets and labels are the harness's, not the paper's. | Rule R1. | Unknown. | F4 measures the combined effect on the paper's setup. |
| D3 | k is selected on the inner split by AUROC. The paper selects the "highest efficacy" result, with no validation split in the code. | Nothing may be fitted on test (M7). | Against LapEigvals, if the paper selected on test. | Report the native cell at each fixed k as well. |
| D4 | The probe seed is the harness seed (0–4), not 42. | The harness seed drives probe initialisation. | Neutral. It affects only the randomised PCA solver. | — |
| D5 | scikit-learn 1.9.0, not 1.6.1. | Project stack. | Unknown, probably neutral. | Run the reader under 1.6.1 on one fold (`uv run --with scikit-learn==1.6.1`). |
| D6 | The harness splits are grouped 5×5 nested CV, not one 80/20 split. | Rule R1. | Neutral. The variance estimate is better. | — |

## Fidelity report

`uv run python src/halluc/methods/lapeigvals/tests/test_fidelity.py`: **10/10 passed**
(2026-10-05, CPU, synthetic attention).

| Test | Result | Notes |
|---|---|---|
| Upstream identity | Pass | Commit and file hashes verified. A wrong hash is refused. |
| F1 | Pass, **bitwise** | Official pipeline (`stack_attention_matrix` → `remove_padding…` → diagonals → top-k) against the adapter, for the three end-of-generation cases, and for T″ > 100 (all k). |
| F1-real | **Pass, bitwise, 20/20** | Job 5307077 (2026-10-06), Llama-3.2-3B-Instruct, 5 items per dataset, shared trace. The official pipeline on the native `generate()` view equals the adapter bitwise. F3: a second generation gives the same ids and identical blocks (20/20). F3b: a generation that asks for other outputs (hidden states only) gives the same ids (20/20). The earlier measurement (0.03, re-forward input) is deviation D1, now removed. |
| F2 | Pass | Shapes, dtype, NaN only after T″. |
| F3 | Pass | Deterministic on CPU. The input tensors are not modified. |
| F4 | **To do** | Llama-3.2-3B-Instruct is one of the paper's models and one of our generators. Paper, temperature 1.0, AUROC: TriviaQA 0.832, NQ-Open 0.693, SQuAD v2 0.757, CoQA 0.812. **These numbers were extracted with a tool from the arXiv HTML. Check them against the PDF before use.** Our setup differs (greedy decoding, different prompt and judges), so compare the direction and the size, not the exact value. |
| F5 | To do | Orientation: inner AUROC > 0.5. Needs the real blocks. |
| F6 | Pass | The extractor has no label input. The reader gives the same probabilities as the official function with the real test labels, so the placeholder labels have no effect. |
| F7 | Pass by construction | No existing detector or stage-3 code path changed. `ForwardTrace` received one optional field. |

## Our earlier reimplementation against the official method

These differences existed in `features/spectral.py` and `detectors.py` before this
adaptation:

| Item | Earlier reimplementation | Official |
|---|---|---|
| Laplacian dtype | float32 | bf16, then float32 |
| Divisor of d_ii | Count of non-zero entries in the column | T − i, always |
| Last token | Always included | Excluded when generation hit the length cap |
| k | 10 only | {5, 10, 25, 50, 100}, filtered |
| Standardisation before PCA | Yes | **No** |
| Decision | Harness threshold | argmax (0.5) |

Measured on synthetic attention (k = 10): the largest absolute difference in the features
is 9.1e-4, for values up to 0.26. This comes from the dtype. On real data, the divisor
can also differ when attention underflows to 0, and for the Gemma sliding-window layers.
The standardisation is the larger difference. On toy data with unequal feature scales,
the two readers give AUROC 0.77 (with the scaler) and 0.51 (official). That toy result
only shows the mechanism. It does not predict the effect on real LapEigvals features,
because all eigenvalues are in [−1, 1].

## Cost
- Extraction: one eager forward per item, plus the official functions on CPU. Not yet
  measured. Use `--limit` for a timing run.
- Storage: about 2 × L·H·100 × 4 bytes per item before compression (1.28 MB for
  Qwen3-14B). The values are bf16 values stored as float32, so they compress well.
- Training: one PCA(512) + logistic regression per fit, on the CPU.
