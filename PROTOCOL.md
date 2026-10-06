# A Protocol to Audit Comparisons Between Hallucination Detectors

Status: draft v0, branch `methodological-issues`. This document is a companion to
`DESIGN.md`. `DESIGN.md` specifies the pipeline. This document specifies how to test if a
comparison between detectors is valid. It also specifies how to measure the error that an
invalid comparison causes.

## 1. The claim that this protocol tests

> Papers that compare white-box hallucination detectors change many things at the same
> time. They change the representation, the classifier, the tuning budget and the
> evaluation. Then they give all of the difference to the representation. When we make
> these axes equal, the ranking changes. Most of the reported gains become smaller or
> disappear.

The evidence comes from this repo. SAPLMA reads the hidden state of the last token. When a
PCA + logistic regression probe reads this state, the result is equal to or better than
every other detector. It is also equal to or better than every combination that we tried.
The SAPLMA paper does not use this probe. Thus the probe causes the difference. A
comparison that uses only the published probe of each method cannot find this.

To make this a general claim, the protocol gives four things:

1. A **decomposition** of a "method". Each confound gets a name (§2).
2. A **list of methodological problems**. Each problem has a test that you can run (§3).
3. Two **procedures**. One procedure tests methods again with data (§4). The other
   procedure audits papers from their text only (§6).
4. **Effect sizes**. They show how much each problem changes the conclusion (§5).

The protocol must also apply to our own result, SAPLMA + PCA+LR (§7). A protocol that
finds errors only in the work of other people is not a valid protocol.

## 2. Decomposition: the parts of a detector

A published detector is a set of six choices. A comparison is valid only on the axes that
it keeps the same.

| Axis | Symbol | What it is | Examples in this repo |
|---|---|---|---|
| Representation | **R** | The internal signal that the method reads: hidden state, attention or logits. Also the layer, the tokens and the feature transform. | SAPLMA, last token at layer ℓ. LapEigvals, top-k Laplacian spectra. ICR, JSD scores. CHARM, attention graph. |
| Reduction | **T** | The projection before the probe. It can use the label or not. | None. PCA-64/128/256. PLS-4/8. |
| Reader | **C** | The classifier or network that changes features into a score. | Logistic regression. MLP (256,128,64). SVM. GNN. |
| Budget | **B** | What the method searches, and how much. This includes the hyperparameter grid, the layer, the number of configurations, the data and early stopping. | SAPLMA searches 7 layer depths. LapEigvals uses a fixed k=10. |
| Decision rule | **D** | How a score becomes a decision. | Threshold that gives the best MCC on an inner split. Threshold 0.5. Threshold tuned on the test set. |
| Evaluation | **E** | Data, splits, groups, metric, seeds, labels and generators. | Grouped nested CV, 5 seeds × 5 folds. MCC and AUROC. Majority of 3 judges. |

A paper can say "R₁ is better than R₂" only if T, C, B, D and E stay the same. As an
alternative, it must optimise each axis equally for both methods. Usually, papers run each
method with its published set of choices. They compare (R₁,T₁,C₁,B₁,D₁) with
(R₂,T₂,C₂,B₂,D₂). Thus they compare two sets of choices, not two representations.

**Main rule:** Each reported difference must come from one axis only. If it does not,
report it as a difference between two sets of choices.

## 3. List of methodological problems

For each problem, this section gives:
- the problem,
- the test that finds it,
- the evidence in this repo.

### M1 — Reader confound
The paper scores each method with its own published probe. Thus it compares the
representation and the classifier together.
- **Test:** Make a grid of representations × readers (§4, step 3). For each
  representation, show the published cell and the best cell. Select the best cell on the
  inner split.
- **Evidence:** For SAPLMA, a change from MLP to PCA+LR increases pooled MCC by **+0.025
  to +0.065** on four generators. On Qwen3-14B, MCC goes from 0.483 to 0.548. This change
  is as large as the difference *between methods*. On Qwen3-14B, SAPLMA-MLP is better than
  LapEigvals by only 0.015. On gemma-3-12b, the difference is 0.014. Scripts:
  `saplma_pcalr_metrics.py`, `reduction_sweep.py`, `fair_comparison.py`.

### M2 — Unequal tuning budget
The paper tunes the hyperparameters of one method. The baselines use default values, or
values from a different setup.
- **Test:** Count the configurations that each method selects from. Record what each
  method searches: layer, width, regularisation, architecture. Run all methods again with
  the same configuration space (§4, step 4).
- **Evidence:** Our own stage-3 protocol had this problem. SAPLMA used four classifiers.
  LapEigvals used a fixed PCA-512 and C=1.0. ICR used only its own MLP. We wrote
  `fair_comparison.py` to remove this problem. `saplma_mlp_tuned.py` removes the opposite
  problem: a tuned logistic regression against an MLP that is not tuned.

### M3 — Missing or weak baselines
The paper compares the new method with other complex methods. It does not compare it with
the simplest signal that can explain the result.
- **Test:** Each comparison must include these baselines:
  1. the majority class,
  2. a surface-feature floor: a probe on layer 0 (the embeddings), and the answer length,
  3. token log-probability features,
  4. the best single representation with its best reader.
- **Evidence:** The layer-0 floor is in `layer_ablation.py`. The 14 log-prob features were
  in `extract_logprob.py` (removed on 2026-10-06; the script is in git history). We also did a length-confound analysis. Log-prob is the only
  representation that is partly different from SAPLMA. Its rescue ratio is 0.66×. Its
  accuracy predicts 0.52×.

### M4 — Combined ablations
The paper gives the gain to one component. But the paper added that component together
with a different change.
- **Test:** For each component that the paper gives credit to, make a pair of methods. The
  two methods must differ **only** in that component.
- **Evidence:** Our former `attn_baseline` (removed on 2026-10-06) and `lapeigvals`
  differed in two things: the Laplacian and PCA. The official AttnEigvals block replaces
  it: it differs from LapEigvals only in the Laplacian. CHARM and the probes differ in three things: the
  graph structure, the capacity and the end-to-end training. The CHARM paper has a control,
  `CHARM (no g.)`. We did not run this control.

### M5 — Settings copied without a test
The paper copies a setting from earlier work or from a different model. It does not test
the setting on the setup that it reports.
- **Test:** Do a sweep of each copied setting. Report how sharp the optimum is. If the
  curve is flat, the result does not depend on the setting. If the curve is sharp, the
  comparison depends on the setting.
- **Evidence:**
  - SAPLMA depth: `layer_ablation.py`.
  - PCA width: `saplma_pca_sweep.py`. PCA-128 came from the budget for the union of
    methods. Nobody tested it for SAPLMA alone.
  - LapEigvals k: `k_sweep.py`.
  - CHARM activation depth: fixed at 0.7, not tested.

### M6 — Combination claims without the correct comparator
The paper says "A and B together are better than A". But the combination also changed the
reader, the reduction or the budget.
- **Test:**
  1. Compare the combination with the best single representation. Use **the same reader
     as the combination**.
  2. Find the items that the combination gets right and the single method gets wrong.
     Compare them with the items that a reader change alone gets right
     (`win_overlap.py`).
  3. Compare each rescue rate with its independence null.
  4. Report the oracle bound as a ceiling. Do not report it as a target.
- **Evidence:** On Qwen3-14B, `union_equal` (0.534) is better than SAPLMA-MLP (0.483).
  But it is worse than SAPLMA + PCA+LR (0.548). Thus the reader causes all of the "union
  gain". Each method rescues the errors of the others at only 0.42–0.58× the rate that
  independence predicts.

### M7 — Leakage in selection and decision
The paper fits something on the data that it uses to compute the score.
- **Test:** For each fitted object, find the split that it was fitted on. Fitted objects
  include the scaler, PCA or PLS, the hyperparameters, the threshold, early stopping and
  the layer choice. Make sure that a group never goes into two splits. An example of a
  group is the questions that share a passage.
- **Evidence:**
  - We found a threshold leak. The code took the mean of the best thresholds of the
    folds. Then it applied this threshold to all items. This increased pooled MCC by
    **+0.0035** on all six generators. Commit `fb5d473` repairs it.
  - CoQA turns share a story. SQuAD questions share a paragraph. Folds without groups put
    near-duplicates in the training split and the test split.
  - If you fit PLS outside the fold, labels leak into the representation.

### M8 — An evaluation that is chosen to give a better result
The metric, the threshold or the pooling changes the ranking. The paper reports only one
of these choices.
- **Test:** Always report these items:
  - a metric without a threshold (AUROC),
  - a metric with a threshold (MCC), with the threshold tuned on an inner split,
  - per-dataset results and pooled results,
  - the mean ± sd over seeds,
  - paired tests on the same seeds and folds,
  - balanced accuracy and the majority-class floor, each time you report accuracy.
- **Evidence:** Base rates go from 0.148 to 0.657 across slices. Thus a constant
  predictor gets an accuracy from 66% to 85% (`full_metrics.py`).

### M9 — Label validity
Some detector "errors" can be label errors. One judge, or judges from the same model
family, can bias the reference labels.
- **Test:**
  1. Use judges from three model families or more.
  2. Report the κ between judges.
  3. Remove INVALID and tied labels from training and from evaluation.
  4. Do a blind audit of a sample of detector errors. Show the results by judge
     agreement.
- **Evidence:** `label_audit_analyze.py`. This audit has a limit: the adjudicator is also
  an LLM. Thus the audit gives a bound on label noise. It does not measure it.

### M10 — Scope: one generator, one domain
The paper measures a ranking on one model, or on instruction-tuned models only. Then it
says that the ranking is a property of the methods.
- **Test:** Use many model families and sizes. Use base models and instruction-tuned
  models. Report per-dataset results and a cross-dataset transfer matrix.
- **Evidence:** With published probes, LapEigvals is **better** than SAPLMA on the two
  base models: llama3.2-3b-base, 0.460 against 0.442; gemma3-12b-pt, 0.454 against 0.430.
  SAPLMA is better on the four instruction-tuned models. With one generator only, a paper
  could have published either ranking.

### M11 — Unequal data and fit budget
Methods see different quantities of training data. Or one method gets early stopping,
restarts or an architecture search, and the probes do not.
- **Test:** For each method, report the number of training rows and the number of fits.
  Use learning curves to show if a gap stays when the data quantity changes.
- **Evidence:** CHARM trains on the inner split only, so it sees less data. The MLP arm
  cannot use `class_weight`, so it has a disadvantage on CoQA. See `learning_curve.py`.

## 4. Procedure to test methods again

Do the steps in this sequence. Each step names the problems that it covers and the output
that it gives.

**Step 0. Lock the evaluation setup (M7, M8).** Before you run a method, fix and record
these items:
- the item pool and `pool_seed`,
- a balanced draw for each seed,
- grouped outer folds, stratified on (dataset, label),
- an inner split for each fitted choice,
- the threshold rule,
- the metrics: AUROC, MCC, balanced accuracy and the majority floor,
- the seeds,
- the label pipeline.

Each method, arm and script must use this setup without changes. See `DESIGN.md`, section
Evaluation.

**Step 1. Decompose (all problems).** For each method, write its published set of
choices (R, T, C, B, D). Get them from the paper and the code. Some choices are not in the
paper, for example the pooling operator of ICR. Treat each of these choices as a
hyperparameter to search. Do not choose a value freely.

**Step 2. Reproduce (check).** Run each method with its published choices. Make sure that
the published *direction* of the results is the same on a comparable setup. The exact
numbers will not be the same on a different generator. If the direction is different,
stop. Find the cause before you continue, because all other steps compare with this
result.

**Step 3. Grid of representations × readers (M1).** Combine each representation with
each reader. Use all published probes **and** a shared set of readers: (none, PCA-k or
PLS-k) × (logistic regression or MLP). For each cell, select on the inner split only. The
output is a matrix. The rows are representations and the columns are readers. Make one
matrix for each metric, generator and dataset. Use `reduction_sweep.py` as the template.

**Step 4. Equal budget (M2, M11).** Give each representation the same configuration
space. Select with the same criterion on the inner split. For each method, record:
- the number of configurations,
- the axes searched,
- the number of training rows,
- the number of fits.

Some differences cannot be removed. An example is `class_weight` in the sklearn MLP. List
each of these differences and the direction of its bias. Use `fair_comparison.py` as the
template.

**Step 5. Baseline floor (M3).** Add these baselines: the majority class, layer 0 and
length, token log-prob, and the best single representation with its best reader. If a
method is not better than the log-prob baseline, do not report it as a white-box gain.

**Step 6. Ablations of one axis (M4).** For each component that a paper gives credit to,
make a pair that differs only in that component. Run the pair with the equal budget of
step 4. Two pairs are missing now:
- the Laplacian with a fixed PCA,
- CHARM without the graph.

**Step 7. Sensitivity of copied settings (M5).** Do a sweep of layer, reduction width, k
and activation depth. Report the peak. Report the width of the plateau: the number of
settings that are within 0.005 AUROC of the peak. If the best value is different from the
copied value, do step 4 again with the best value.

**Step 8. Combination claims (M6).** A combination is a gain only if it is better than the
best single representation **with the same reader and budget**. A paired test must show
this. Also report:
- the item-level overlap with the gain from a reader change alone,
- the rescue rates and their independence null,
- the oracle ceiling.

**Step 9. Scope (M10).** Do steps 2 to 8 again on these setups:
- many generator families and sizes,
- base models and instruction-tuned models,
- each dataset,
- cross-dataset transfer.

A claim is true only if it is true in all cells. If it is not, state the cells where it
is false.

**Step 10. Labels (M9).** Report the agreement between judges. Do a blind audit of
detector errors. Make sure that the method ranking stays the same on the subset where all
judges agree.

**Step 11. Statistics (M8).** For each pairwise claim, test the paired difference across
seeds × folds. As an alternative, use a paired bootstrap over items in each seed. Compare
the effect with the sd across seeds. Correct for the number of comparisons in the grid.

## 5. How to measure the effect of a problem

The paper reports these quantities. Each quantity changes a methodological problem into an
effect size.

| Measure | Definition | What it shows |
|---|---|---|
| **Reader gap** Δ_C(R) | score(R, best reader) − score(R, published reader) | How much the published probe made a method look worse (or better). |
| **Budget gap** Δ_B(R) | score with equal budget − score with published budget | How much of a published gain came from more search. |
| **Confound share** | Δ_C(R) ÷ published gap between R and its comparator | The part of a claimed gain that the reader alone explains. |
| **Ranking stability** | Kendall τ between the ranking with published choices and the ranking with equal choices. Also, the number of pairs whose order changes. | If the conclusion of the field stays true when the axes are equal. |
| **Claim survival rate** | The fraction of published claims "A > B" that stay significant with equal choices, across generators and datasets. | The main number of the audit. |
| **Gain attribution** | The union gain, divided into a reader part and a block part, from the item-level overlap. | If a combination adds signal, or only changes the reader. |
| **Scope consistency** | The fraction of (generator, dataset) cells where a claim is true. | If a ranking is general, or true for one setup only. |

Example on Qwen3-14B, pooled MCC:
- The union gain over SAPLMA with its published probe is +0.051 (0.534 − 0.483).
- The reader gap of SAPLMA is +0.065 (0.548 − 0.483).
- Thus the confound share is more than 1. The reader change alone explains more than all
  of the gain.

## 6. Procedure to audit papers

This procedure applies the list of problems to published papers. It does not run the
methods again. It changes "most papers do this" from an impression into a measured claim.

1. **Corpus.** Make a list of white-box hallucination-detection papers that compare
   detectors. Fix the inclusion criteria before you start: venues, years, and "the paper
   proposes a detector and compares it with two or more other detectors". Record the
   search strings and the dates.
2. **Coding sheet.** For each paper and each comparison table, code each item as yes, no
   or unclear. Give a page or table reference.

   | Item | Problem |
   |---|---|
   | All methods use the same reader, or each method gets a reader search. | M1 |
   | The paper reports a hyperparameter search for the baselines, not only for the new method. | M2 |
   | The paper reports the number of configurations for each method. | M2 |
   | The paper includes log-prob or a different confidence baseline. | M3 |
   | The paper does an ablation of each claimed component alone. | M4 |
   | A sweep on the reported setup justifies the layer, width and k choices. | M5 |
   | The paper compares a combination with the best single method, with the same reader. | M6 |
   | Model selection and threshold use held-out data. Splits use groups where necessary. | M7 |
   | The paper reports metrics with and without a threshold, seeds, variance and significance tests. | M8 |
   | The paper describes the label source, reports agreement and does an audit. | M9 |
   | The paper uses more than one generator, with base models and instruction-tuned models. | M10 |
   | Training data and fit budget are equal for all methods. | M11 |

3. **Double coding.** Two coders code a subset. Report their κ. Write a rule to resolve
   disagreements.
4. **Link to the experiments.** For the methods that we ran again, connect the problems
   found in each paper to the Δ values measured in §5. The argument of the paper depends on
   this link. The audit shows that the problem is frequent. The experiments show that the
   problem is large where we measured it.

## 7. Apply the protocol to our own claim

"SAPLMA + PCA+LR is better than all other methods" is also a comparison. Thus the protocol
must test it. These points are open:

- **Selection bias in our favour.** We chose PCA+LR as the reference reader because it
  won on SAPLMA. The reader grid (step 3) must show two things. First, the other
  representations got the same set of readers. Second, they lost on merit. It must not
  show that we chose the reader for SAPLMA only. `fair_comparison.py` covers this for
  linear readers and MLP readers. Use its numbers as the main result. Do not use the table
  with published probes as the main result.
- **Base generators.** With published probes, LapEigvals is better than SAPLMA on the two
  base models. Find out if SAPLMA + PCA+LR is still better **than LapEigvals with its own
  best reader** on these models. If it is not, change the claim. Say that it is true for
  instruction-tuned models only.
- **Results that are too close.** On gemma-3-12b, `union_equal` (0.575) and SAPLMA +
  PCA+LR (0.571) are within noise. Change "is better than" to "is not beaten by". Use a
  paired test to decide.
- **CHARM.** Our reimplementation was removed on 2026-10-06. It did not run on the gemma
  generators (fp16 overflow), it had no ablation without the graph, and it trained on
  less data. Adapt the official code (`Noired/charm`) and do the ablation without the
  graph. Until then, there is no valid CHARM comparison (M4, M11).
- **Other disadvantages.** In `fair_comparison`, the SAPLMA layer is fixed and not tuned.
  This is a disadvantage for our claim. The MLP arm has no class weights. This is a
  disadvantage for non-linear readers. List both, with the direction of each bias.

## 8. What this repo covers now

| Step | Status | Where |
|---|---|---|
| 0 Evaluation setup | Done. | stages 1–3, `DESIGN.md` |
| 1–2 Decompose and reproduce | LapEigvals adapted from the official code. SAPLMA and ICR are reimplementations. CHARM: to adapt (reimplementation removed). | stage 3, `src/halluc/methods/` |
| 3 Reader grid | Done for 5 readers × 6 methods. Must be put into one table. | `reduction_sweep.py` |
| 4 Equal budget | Done. The budget record for each method is not in a table yet. | `fair_comparison.py`, `saplma_mlp_tuned.py` |
| 5 Baseline floor | Done for layer 0 and length. Log-prob was done, then removed on 2026-10-06; add it again to meet M3. | `layer_ablation.py` |
| 6 Ablations of one axis | Laplacian: possible now (official LapEigvals against AttnEigvals, same reader). **Missing**: CHARM without the graph. | `run_grid.py` |
| 7 Sensitivity | Done for SAPLMA layer, PCA width and k. | `layer_ablation.py`, `saplma_pca_sweep.py`, `k_sweep.py` |
| 8 Combinations | Done, with many methods. | `win_overlap.py`, `vote*.py`, `union_*.py`, `oracle_bound.py` |
| 9 Scope | 6 generators. Transfer done for SAPLMA only. | `dataset_transfer.py` |
| 10 Labels | Audit by an LLM adjudicator. No human audit. | `label_audit_analyze.py` |
| 11 Statistics | Per-seed values are stored. Paired tests and multiplicity correction are **not systematic yet**. | — |
| §5 Measures | **Not computed yet** as this document defines them. | — |
| §6 Paper audit | **Not started.** | — |

Two parts change this work from a study of five methods into a methodological
contribution:
- the §5 measures, computed on the full grid,
- the §6 paper audit.

Both parts are not done yet.
