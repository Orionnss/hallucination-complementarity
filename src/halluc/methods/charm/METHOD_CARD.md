# CHARM

## Source
- Paper: Frasca et al., *Neural Message-Passing on Attention Graphs for Hallucination
  Detection*, ICLR 2026, arXiv 2509.24770
- Code: https://github.com/Noired/charm @ `7ab3ab7`, MIT (SOURCE.md)
- Status: **adapted**. The adapter calls the official functions from an unchanged
  checkout. Our earlier reimplementation (stage 6) was removed on 2026-10-06.

## Type
B — end-to-end trained (ADDING_A_METHOD.md §3).

## Pipeline

Upstream paths are relative to `original-repos/charm`.

| Part | Paper | Code | Our adapter |
|---|---|---|---|
| Input | Attention maps and activations of the generation. | `model.generate(output_attentions=True, output_hidden_states=True, return_dict_in_generate=True)` (`data_prep_2/data_collection_utils.py:200`, `data_prep_2/data_collection.py:57`). | **Harness, same call.** The native view of the shared trace, cut to the kept answer, moved to the CPU (upstream moves each slice there itself). |
| Graph | Tokens are nodes; an edge (i, j) carries the attention of i to j over all layers and heads; sparsified at τ = 0.05 (Eq. 1); node features are the self-attention. | `dataset/graphs.py: get_data_object(attentions, threshold=0.05, prompt_graph=False)` (README); x and edge_attr cast to float16 (`data_collection.py:76-77`). `prompt_len = P − 1`: the last prompt token predicts the first answer token. | Called unchanged; stored per item. |
| Activations | Residual-stream activations of one layer as node attributes. | `dataset/acts.py: get_activations(hidden_states, layers)`; README collects 24, 28, 32; configs train on `act_llm_layers: "24"`; float16. | Called unchanged for one layer at 0.75 of the depth (= 24 of 32). See D2. |
| Transforms | — | `training.get_transforms(args)`: LabelPooling, ThresholdAttention(0.05), MarkPromptEdges for this config. | Called unchanged on each graph. |
| Reader | GNN with message passing; mean readout over response nodes. | `model/mp.py: CHARM`, built by `training.get_model`, trained by `training.train_model`: BCEWithLogits, Adam, plateau scheduler on the validation AUPR of the hallucination class, early stop, best-epoch checkpoint. Config `configs/charm/movies_att_act.yaml`: hidden 32, 1 layer, dropout 0.25, lr 1e-3, 50 epochs, patience 20, batch 32, weight decay 0.1 on the activation encoder. | `reader.OfficialCharm` runs these functions unchanged on the same objects. |
| Decision | Threshold-free metrics (AUROC, AUPR). | — | The harness threshold. A 0.5 cut is recorded as the native decision. |
| Split | Train/validation/test 60/20/20, or a provided test set with validation carved from training. | `data_prep_*/data_split.py`. | **Harness**: grouped nested CV. The reader takes its 20% validation set from its own training rows. |
| Labels | Correctness (1 = correct answer). | `evaluate`: AUPR(hallucination) = AP(1 − y, −logit). | The reader trains on 1 − y and returns 1 − sigmoid(logit) as the hallucination probability. |

### Choice of configuration
The repository has 10 configurations: 5 datasets × {attention only, attention +
activations}. `movies_att_act` is used, because it is the closest to our task: one
label per answer (`tokenwise: false`), short factual QA, attention + activations. The
NQ and CNN configurations are token-level (span annotations), which our labels are not.

## Feature blocks (ragged, one dict per item)

| Block | Shape per item | dtype | Content |
|---|---|---|---|
| `charm_x` | [N, L·H] | float16 | Node features (self-attention per layer and head). |
| `charm_edge_index` | [2, E] | int32 (int64 upstream; lossless) | Edges. |
| `charm_edge_attr` | [E, L·H] | float16 | Edge features. |
| `charm_act` | [N, d] | float16 | Activations of the chosen layer. |
| `charm_prompt_len` | [1] | int32 | Upstream `prompt_len` (P − 1). |
| `charm_act_layer`, `charm_n_heads` | [1] | int32 | To rebuild `Data.head` / `Data.layer`. |

N = number of tokens read (prompt + kept answer). Store: `runs/<run>/methods/charm_official/<dataset>/`.

## Native reader
- Hyperparameters searched: none (the configuration is fixed).
- Number of configurations: 1.

## Parts of the source code not used

| Part | Why |
|---|---|
| `data_prep_1/*`, `transformers-4.32.0/` | NQ/CNN preparation with teacher forcing from Lookback Lens data. The harness generates. |
| `data_prep_2/data_collection.py` loop, `data_annotation.py`, `data_split.py`, `data_consolidation.py` | Generation loop, labelling and splits: the harness owns them. The two extraction functions it calls are used. |
| `dataset/dataset.py` (AttentionDataset), `run_exp.py` | Dataset directories on disk; the reader builds the same `Data` objects in memory (test F1). |
| `baseline/*`, `evaluation.py`, `checkpoints/` | Other baselines and released checkpoints. |
| `scores` / `atps` | Token-probability artefacts used only by baselines. |

## Patches
None. `torch_geometric` is pinned to 2.6.1 instead (SOURCE.md).

## Deviations

| Id | What | Why | Direction | Test |
|---|---|---|---|---|
| D1 | One fixed configuration (`movies_att_act`), not a search over the paper's space (Table 9). | Budget; the repository releases fixed configurations. | Unknown, probably against CHARM. | Run a second configuration on one seed. |
| D2 | Activation layer at 0.75 of the depth (24 of 32 in the paper's models). | Our generators have other depths. | Unknown. The paper reports robustness to this choice (Table 7). | — |
| D3 | The validation split comes from the training rows, stratified on the label, not grouped. | The reader does not see the groups. | Small. | — |
| D4 | Generation prompt, generators, datasets and labels are the harness's. | Rule R1. | Unknown. | F4. |
| D5 | Data loaders run in the main process. | Memory: worker processes would copy the graphs. | None. | — |

## Fidelity report

`uv run python src/halluc/methods/charm/tests/test_fidelity.py`: **6/6 passed**
(2026-10-06, CPU, synthetic `generate()` outputs).

| Test | Result | Notes |
|---|---|---|
| Upstream identity | Pass | Commit and file hashes verified. |
| F1 | Pass, **bitwise** | The blocks equal `get_data_object` + `get_activations` + float16 casts on the same outputs (3 end-of-generation cases). The reader's `Data` equals upstream's `Data` + `hydrate`, field by field. |
| F1-real | To do | GPU. |
| F3 | Pass | Deterministic; the trace is not modified. |
| F4, F5 | To do | After the full rerun. |
| F6 | Pass | The extractor has no label input. The reader trains the upstream `CHARM` (module `model.mp`) and is deterministic for a given seed. |

## Cost
- Extraction: on the CPU inside stage 1 (upstream builds one dense T×T matrix per layer
  and head). Measured in the stage-1 GPU test.
- Storage: dense float16 edge features [E, L·H]; the largest block of the study.
- Training: GNN; uses a GPU when one is available (`train_model`).
