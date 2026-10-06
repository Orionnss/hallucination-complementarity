# ICR Probe

## Source
- Paper: Zhang et al., *ICR Probe: Tracking Hidden State Dynamics for Reliable
  Hallucination Detection in LLMs*, ACL 2025, arXiv 2507.16488
- Code: https://github.com/XavierZhang2002/ICR_Probe @ `40ec490`, Apache-2.0 (SOURCE.md)
- Status: **adapted**. The adapter calls the official functions from an unchanged checkout.

## Type
A — supervised probe (ADDING_A_METHOD.md §3).

## Pipeline

Upstream paths are relative to `original-repos/icr_probe`.

| Part | Paper | Code | Our adapter |
|---|---|---|---|
| Input | Hidden states and attentions of the generation. | `ICRScore(hidden_states, attentions, ...)` reads the `generate()` outputs: per step, per layer (README, "1. Compute ICR Scores"; `src/icr_score.py:20`). | **Harness, same call.** The native view of the shared trace, cut to the kept answer. |
| Preprocessing | — | `_pre_process_hs`, `_pre_process_attn` stack the steps into full matrices; `set_other_attn_scores_to_zero` keeps attention only inside the user prompt and inside the response (`core_positions`, `src/icr_score.py:104`). | Called unchanged. `core_positions` come from the harness (see below). |
| Feature computation | ICR score of a token at a layer = JSD between the projection of the residual update on the attended tokens and the attention over them (§3). | `compute_icr(top_k=20, top_p=0.1, pooling='mean', attention_uniform=False, hidden_uniform=False, use_induction_head=True)` with `skew_threshold=0, entropy_threshold=1e5` (README). top_p overrides top_k (`src/icr_score.py:226`). | Called with the README values. |
| Pooling to one vector | "Pooled ICR Score" (§4.2). | The mean over response tokens per layer: `np.mean(item, axis=-1)` (`scripts/empirical_study.ipynb`, `read_acd_scores`). | The same mean. Block `icr_official` = `[L]`. This resolves the open item of our reimplementation (DESIGN.md). |
| Reader | MLP (L, 128, 64, 32, 1) (§4). | `src/utils.py: ICRProbe` (batch norm, dropout 0.3, leaky ReLU, sigmoid); `src/icr_probe.py: ICRProbeTrainer` (BCE, Adam lr 1e-3, weight decay 1e-5, ReduceLROnPlateau 0.5/5 on validation loss, 100 epochs, batch 16, best epoch by validation loss); `src/config.py`. | `reader.OfficialIcrProbe` runs `setup_model`, `train`, `_train_epoch`, `_validate_epoch` and `save_model` unchanged, and gives them the data loaders that the upstream trainer does not have (see "Parts not runnable upstream"). |
| Decision | — | `outputs.round()` in `_validate_epoch` (threshold 0.5). | Recorded as `native_preds__…`. The main results use the harness threshold. |
| Split | — | `Config.test_size = 0.2` (validation). | **Harness**: grouped nested CV. The reader takes its 20% validation set from its own training rows. |

### `core_positions`

The upstream repository does not show how its authors set `user_prompt_start`,
`user_prompt_end` and `response_start`. The adapter uses:
- `user_prompt_start`, `user_prompt_end`: the token span of the item's own text in the
  prompt (`ForwardTrace.user_span`; the user message in a chat template, the final
  question block in a base model's few-shot prompt). Tested on the Qwen3, Llama-3.2
  (instruct and base) and Gemma-3 tokenizers: the span decodes to exactly the item text.
- `response_start`: the first generated token (`prompt_len`).

These positions decide which attention rows survive `set_other_attn_scores_to_zero`, so
they change the induction-head choice (`_is_induction_head`). See D1.

## Feature blocks

| Block | Type | Shape per item | dtype | Content |
|---|---|---|---|---|
| `icr_official` | vector | [L] | float32 | Mean ICR score per layer over the response tokens. NaN for an empty answer. |
| `icr_official_ntok` | scalar | [1] | int32 | Number of response tokens that were scored. |

Store: `runs/<run>/methods/icr_official/<dataset>/`.

## Native reader
- Hyperparameters searched: none.
- Hyperparameters fixed: the `Config` defaults above.
- Number of configurations: 1.

## Parts not runnable upstream

| Part | Problem | What the reader does |
|---|---|---|
| `ICRProbeTrainer.setup_data` | Calls `_load_data` and `_create_data_loaders`, which do not exist. | Builds the loaders: 80/20 stratified split of the training rows; training batches of 16, shuffled, last incomplete batch dropped (BatchNorm cannot train on one row); validation in one batch (eval-mode BatchNorm makes the batching irrelevant). |
| `_validate_epoch` | Reads `config.halu_threshold`, which `Config` does not define. | Patch P2. |
| `ICRProbeTrainer.__init__` | The README calls it without `model`; the code requires it. | Passes `model=None`; `setup_model` creates the model. |

## Parts of the source code not used

| Part | Why |
|---|---|
| `scripts/empirical_study.ipynb` (except the pooling rule) | Plots and analyses. |
| ICR ablation switches (`attention_uniform`, `hidden_uniform`) | Ablations; the README value (False) is used. |

## Patches

| Id | What | Why | Effect on the computation |
|---|---|---|---|
| P1 | In `src/icr_score.py` only, `torch` is a shim whose `cuda.device(...)` is a no-op context for a CPU device and whose `cuda.empty_cache()` is skipped when CUDA is absent. Every other attribute is the real torch. | Device placement: upstream wraps cache releases in `torch.cuda.device(icr_device)`, which raises on CPU. Needed for the CPU tests. | None. On a CUDA device every call is the real one. |
| P2 | `config.halu_threshold = 0.5` on the config object. | Upstream reads an attribute that its `Config` does not define. | None. It feeds only the logged precision and recall, not the fit or the epoch choice. |

## Deviations

| Id | What | Why | Direction | Test |
|---|---|---|---|---|
| D1 | `core_positions` from the harness's user span and the first generated token. | Not specified upstream. | Unknown. | Run with the user span = the whole prompt, on one seed. |
| D2 | Generation prompt, generators, datasets and labels are the harness's. | Rule R1. | Unknown. | F4. |
| D3 | The validation split for the best epoch comes from the training rows, stratified on the label, not grouped. | The reader does not see the groups. | Small: CoQA/SQuAD items that share a passage can fall on both sides of this inner split. | — |
| D4 | An empty answer gives NaN; the reader reads it as 0. | The upstream probe asserts finite input. | Neutral, rare. | Count with `icr_official_ntok == 0`. |

## Fidelity report

`uv run python src/halluc/methods/icr_probe/tests/test_fidelity.py`: **8/8 passed**
(2026-10-06, CPU, synthetic `generate()` outputs).

| Test | Result | Notes |
|---|---|---|
| Upstream identity | Pass | Commit and file hashes verified. |
| F1 | Pass, **bitwise** | Adapter = the README usage called directly on the same outputs, with the notebook's pooling. Core positions change the result as expected. |
| F1-real | To do | GPU. |
| F2 | Pass | `[L]` float32; NaN and ntok = 0 for an empty answer; a trace without `user_span` is refused. |
| F3 | Pass | Deterministic; the trace is not modified. |
| F4 | To do | After the full rerun. |
| F5 | To do | After the full rerun. |
| F6 | Pass | The extractor has no label input. The reader trains the upstream `ICRProbe` (module `src.utils`) and is deterministic for a given seed. |

## Our earlier reimplementation against the official method

`features/icr.py` (still extracted as `icr`) differs from the official code in at least:
no attention masking by `core_positions`; no induction-head selection; top-10 tokens
instead of top 10%; JSD without the official standardisation of both vectors before the
softmax; and two pooling choices (last token, mean) instead of the official mean.

## Cost
- Extraction: on the GPU, inside stage 1. Measured in the stage-1 GPU test.
- Storage: [L] float32 per item.
- Training: one small MLP per fit, 100 epochs.
