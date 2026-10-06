"""Fidelity tests for the shared trace and the LapEigvals adapter (ADDING_A_METHOD.md §7).

CPU only, synthetic `generate()` outputs. They prove two things:

  * the harness cut (ForwardTrace.from_generate) keeps exactly the prompt + kept answer,
    and its stacked view equals the full matrices it was simulated from;
  * the adapter gives the official functions their own input (the native per-step view)
    and returns exactly what the official pipeline returns on it, bitwise.

They say nothing about how well the method detects hallucinations. Real generate()
outputs are covered by f1_real.py (GPU).

Run:  uv run python src/halluc/methods/lapeigvals/tests/test_fidelity.py
  or: uv run --with pytest pytest src/halluc/methods/lapeigvals/tests
"""

from __future__ import annotations

import dataclasses
import inspect
import sys
import traceback

import numpy as np
import torch

from halluc.features.base import ForwardTrace
from halluc.methods.lapeigvals import K_MAX, OfficialSpectralFeatures, load
from halluc.methods.lapeigvals.adapter import ATTN_BLOCK, LAP_BLOCK, LEN_BLOCK
from halluc.methods.lapeigvals.reader import OfficialProbe, native_argmax
from halluc.methods.lapeigvals.upstream_spec import SPEC, TOP_K_EIGVALS
from halluc.methods.upstream import UpstreamMismatch, verify

from halluc.methods.testing import PAD, causal_attention, ids_for, make_case  # noqa: F401


def official_pipeline(steps, generated_tokens: torch.Tensor) -> dict[str, np.ndarray]:
    """The upstream call sequence of feature_storage.py + train_attn_vs_laplacian.py."""
    up = load()
    stacked = up["attention_weights"].stack_attention_matrix(steps)
    (example,) = up["processing"].remove_padding_from_intermediate_states(
        per_layer_batched_data=stacked, data_type="attn",
        generated_tokens=generated_tokens.unsqueeze(0), pad_token_id=PAD,
    )
    attn = up["attention_weights"].attention_diagonal(example).float()
    lap = up["attention_weights"].laplacian_diagonal_from_attn(example, vertical_edges=False).float()
    n_tokens = lap.shape[-1]
    out = {}
    for k in TOP_K_EIGVALS:
        if k <= n_tokens:
            out[f"lap{k}"] = up["attn_feats"].get_laplacian_eigvals_per_head_topk([lap], None, k)[0].numpy()
            out[f"attn{k}"] = up["attn_feats"].get_attn_eigvals_per_head_topk([attn], None, k)[0].numpy()
    out["T"] = n_tokens
    return out


def adapter_views(feats: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    n_tokens = int(feats[LEN_BLOCK][0])
    out = {}
    for k in TOP_K_EIGVALS:
        if k <= n_tokens:
            out[f"lap{k}"] = feats[LAP_BLOCK][..., :k].reshape(-1)
            out[f"attn{k}"] = feats[ATTN_BLOCK][..., :k].reshape(-1)
    out["T"] = n_tokens
    return out


def assert_bitwise(official: dict, adapted: dict) -> None:
    assert official.keys() == adapted.keys(), (official.keys(), adapted.keys())
    for key in official:
        if key == "T":
            assert official[key] == adapted[key], (official[key], adapted[key])
        else:
            assert np.array_equal(official[key], adapted[key]), f"{key} differs"


EXTRACTOR = None


def extractor() -> OfficialSpectralFeatures:
    global EXTRACTOR
    if EXTRACTOR is None:
        EXTRACTOR = OfficialSpectralFeatures()
    return EXTRACTOR


# --------------------------------------------------------------------- harness: the cut

# (P, n generated, keep): stopped on a terminator; hit the length cap; base model trimmed
# after a stop string; empty answer.
CUT_CASES = {"stopped": (9, 8, 7), "length_cap": (9, 12, 12),
             "trimmed": (9, 15, 6), "empty": (9, 1, 0)}


def test_cut_keeps_prompt_and_kept_answer():
    for name, (P, n, keep) in CUT_CASES.items():
        trace, attn, hid, seq = make_case(P, n, keep, L=3, H=2, seed=11)
        t_prime = min(P + keep, P + n - 1)
        assert trace.seq_len == t_prime, (name, trace.seq_len, t_prime)
        assert len(trace.step_attentions) == 1 + t_prime - P, name
        assert len(trace.step_logits) == 1 + t_prime - P, name
        assert torch.equal(trace.input_ids, seq[: t_prime + 1]), name


def test_stacked_view_equals_full_matrices():
    for name, (P, n, keep) in CUT_CASES.items():
        trace, attn, hid, _ = make_case(P, n, keep, L=3, H=2, seed=12)
        T = trace.seq_len
        for layer in range(3):
            assert torch.equal(trace.attentions[layer], attn[layer][:, :T, :T]), (name, layer)
        for layer in range(4):
            assert torch.equal(trace.hidden_states[layer], hid[layer][:T]), (name, layer)


# ------------------------------------------------------------------- adapter: F1 to F6

def test_f1_stopped():
    P, n, keep = 9, 8, 7                      # 7 answer tokens + EOS
    trace, _, _, seq = make_case(P, n, keep, L=4, H=3, seed=1)
    official = official_pipeline(trace.step_attentions, seq)
    assert_bitwise(official, adapter_views(extractor().extract(trace)))
    assert official["T"] == P + keep


def test_f1_length_cap():
    P, n, keep = 9, 12, 12                    # the last answer token was never read
    trace, _, _, seq = make_case(P, n, keep, L=3, H=2, seed=2)
    official = official_pipeline(trace.step_attentions, seq)
    assert_bitwise(official, adapter_views(extractor().extract(trace)))
    assert official["T"] == P + keep - 1


def test_f1_trimmed_base_answer():
    """The harness cut is the input: upstream on the cut steps and ids = the adapter."""
    P, n, keep = 9, 15, 6
    trace, _, _, seq = make_case(P, n, keep, L=3, H=2, seed=3)
    official = official_pipeline(trace.step_attentions, seq[: P + keep + 1])
    assert_bitwise(official, adapter_views(extractor().extract(trace)))
    assert official["T"] == P + keep


def test_f1_long_sequence_all_k():
    """T'' > K_MAX, so every k of the official sweep is offered, and all match."""
    P, n, keep = 60, 71, 70
    trace, _, _, seq = make_case(P, n, keep, L=2, H=2, seed=4)
    official = official_pipeline(trace.step_attentions, seq)
    adapted = adapter_views(extractor().extract(trace))
    assert_bitwise(official, adapted)
    assert all(f"lap{k}" in adapted for k in TOP_K_EIGVALS)


def test_f2_contract():
    """Shapes, dtypes, NaN only past the sequence length, finite elsewhere."""
    P, n, keep, L, H = 9, 8, 7, 4, 3
    trace, _, _, _ = make_case(P, n, keep, L, H, seed=5)
    feats = extractor().extract(trace)
    T = trace.seq_len
    for block in (LAP_BLOCK, ATTN_BLOCK):
        x = feats[block]
        assert x.shape == (L, H, K_MAX) and x.dtype == np.float32, (x.shape, x.dtype)
        assert np.isfinite(x[..., :T]).all()
        assert np.isnan(x[..., T:]).all()
    assert feats[LEN_BLOCK].tolist() == [T]


def test_f2_refuses_trace_without_native_view():
    attn = causal_attention(2, 2, 12, seed=6)
    full = ForwardTrace.from_full(attn, (), prompt_len=5, input_ids=ids_for(13, 6))
    try:
        extractor().extract(full)
    except ValueError:
        return
    raise AssertionError("a trace without the native view was accepted")


def test_f3_determinism():
    trace, _, _, _ = make_case(9, 8, 7, L=4, H=3, seed=7)
    a = extractor().extract(trace)
    b = extractor().extract(trace)
    for key in a:
        assert np.array_equal(a[key], b[key], equal_nan=True), key


def test_f3_input_not_mutated():
    """Upstream helpers write into tensors in place in places; the trace must stay intact."""
    trace, _, _, _ = make_case(9, 8, 7, L=3, H=2, seed=8)
    before = [[a.clone() for a in step] for step in trace.step_attentions]
    extractor().extract(trace)
    for step_before, step_after in zip(before, trace.step_attentions):
        assert all(torch.equal(x, y) for x, y in zip(step_before, step_after))


def test_f6_extractor_sees_no_labels():
    params = set(inspect.signature(OfficialSpectralFeatures.extract).parameters)
    assert params == {"self", "trace"}, params


def _toy_probe_data(seed: int = 0):
    rng = np.random.default_rng(seed)
    n_train, n_test, d = 800, 200, 700
    y = rng.integers(0, 2, n_train + n_test)
    X = rng.normal(size=(n_train + n_test, d)).astype(np.float32)
    X[:, :5] += y[:, None] * 0.8
    return X[:n_train], y[:n_train], X[n_train:], y[n_train:]


def test_f6b_reader_equals_official_and_ignores_test_labels():
    """OfficialProbe = train_logistic_regression with the real test labels."""
    X_tr, y_tr, X_te, y_te = _toy_probe_data()
    ours = OfficialProbe(random_seed=42).fit(X_tr, y_tr).predict_proba(X_te)
    lr = load()["lr"]
    direct = lr.train_logistic_regression(
        features=torch.from_numpy(np.concatenate([X_tr, X_te])),
        labels=torch.from_numpy(np.concatenate([y_tr, y_te])),
        split={"train_idx": torch.arange(len(X_tr)),
               "test_idx": torch.arange(len(X_tr), len(X_tr) + len(X_te))},
        pca_dim=512, use_cuda=False, random_seed=42,
    )
    assert np.array_equal(ours, np.asarray(direct["test_proba"])), "probabilities differ"
    assert np.array_equal(native_argmax(ours[:, 1]), np.asarray(direct["test_preds"]))


def test_upstream_identity():
    """The checkout is the pinned commit, and a changed file hash is refused."""
    verify(SPEC)
    bad = dataclasses.replace(SPEC, files={**SPEC.files,
                                            "hallucinations/probe_models/lr.py": "0" * 64})
    try:
        verify(bad)
    except UpstreamMismatch:
        return
    raise AssertionError("a wrong file hash was accepted")


# ----------------------------------------------------------------- deviation measures
# Not pass/fail: how far our reimplementation sits from the official method on the
# same trace, for METHOD_CARD.md.

def measure_old_extractor_gap() -> dict:
    from halluc.features.spectral import LapEigvals

    trace, _, _, _ = make_case(9, 31, 30, L=4, H=3, seed=9)
    old = LapEigvals(k=10).extract(trace)["lapeigvals"]
    new = extractor().extract(trace)[LAP_BLOCK][..., :10]
    return {"max_abs_diff_k10": float(np.abs(old - new).max()),
            "max_abs_value": float(np.abs(new).max())}


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {name}")
            traceback.print_exc()
    print("\ndeviation, our extractor vs official:", measure_old_extractor_gap())
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
