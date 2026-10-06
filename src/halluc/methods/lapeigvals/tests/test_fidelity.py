"""Fidelity tests F1-F6 for the LapEigvals adapter (ADDING_A_METHOD.md §7).

CPU only, synthetic attention. These tests prove that the adapter feeds the official
functions exactly what the official pipeline would feed them, and that the native reader
returns exactly what the official probe returns. They say nothing about how well the
method detects hallucinations.

The one part they cannot cover is the generator itself: upstream reads attention rows
from incremental decoding with a KV cache, the adapter from one full forward. That
difference is numerical only, and needs a GPU to measure (F1-real, METHOD_CARD.md).

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

PAD, EOS, TERM = 0, 2, 106


def causal_attention(n_layers: int, n_heads: int, T: int, seed: int) -> tuple[torch.Tensor, ...]:
    """Row-stochastic lower-triangular attention, in bf16 like the generator's output."""
    g = torch.Generator().manual_seed(seed)
    mask = torch.ones(T, T).tril().bool()
    layers = []
    for _ in range(n_layers):
        logits = torch.randn(n_heads, T, T, generator=g) * 3.0
        logits = logits.masked_fill(~mask, float("-inf"))
        layers.append(torch.softmax(logits, dim=-1).to(torch.bfloat16))
    return tuple(layers)


def generate_steps(attn_full: tuple[torch.Tensor, ...], prompt_len: int, t_prime: int):
    """What `model.generate(output_attentions=True)` returns at batch size 1.

    Step 0 holds the prompt rows; step j >= 1 holds the single row of the token fed at
    that step, over the j + prompt_len keys seen so far. Rows reach t_prime in total.
    """
    steps = [tuple(a[:, :prompt_len, :prompt_len].unsqueeze(0) for a in attn_full)]
    for row in range(prompt_len, t_prime):
        steps.append(tuple(a[:, row : row + 1, : row + 1].unsqueeze(0) for a in attn_full))
    return tuple(steps)


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


def ids_for(T: int, seed: int, last: int | None = None) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(10, 1000, (T,), generator=g)
    if last is not None:
        ids[-1] = last
    return ids


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
        EXTRACTOR = OfficialSpectralFeatures(pad_token_id=PAD, eos_token_id=EOS)
    return EXTRACTOR


# ------------------------------------------------------------------------------- tests

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


def test_f1_stopped_terminator_stripped():
    """Stage 1 stripped the EOS: upstream had one more token and the same T' = T rows."""
    P, n_ans, L, H = 9, 7, 4, 3
    T = P + n_ans
    attn, ids = causal_attention(L, H, T, seed=1), ids_for(T, seed=1)
    official = official_pipeline(generate_steps(attn, P, T), torch.cat([ids, torch.tensor([EOS])]))
    feats = extractor().extract(ForwardTrace(attn, (), P, ids), stopped=True,
                                ends_with_terminator=False)
    assert_bitwise(official, adapter_views(feats))
    assert official["T"] == T


def test_f1_hit_length_cap():
    """max_new_tokens reached: the last answer token was never fed back, T' = T - 1."""
    P, n_ans, L, H = 9, 12, 3, 2
    T = P + n_ans
    attn, ids = causal_attention(L, H, T, seed=2), ids_for(T, seed=2)
    official = official_pipeline(generate_steps(attn, P, T - 1), ids)
    feats = extractor().extract(ForwardTrace(attn, (), P, ids), stopped=False,
                                ends_with_terminator=False)
    assert_bitwise(official, adapter_views(feats))
    assert official["T"] == T - 1


def test_f1_terminator_kept():
    """Gemma: stage 1 kept <end_of_turn> (106) as the final token, so T' = T - 1."""
    P, n_ans, L, H = 9, 6, 3, 2
    T = P + n_ans
    attn, ids = causal_attention(L, H, T, seed=3), ids_for(T, seed=3, last=TERM)
    official = official_pipeline(generate_steps(attn, P, T - 1), ids)
    feats = extractor().extract(ForwardTrace(attn, (), P, ids), stopped=True,
                                ends_with_terminator=True)
    assert_bitwise(official, adapter_views(feats))


def test_f1_long_sequence_all_k():
    """T'' > K_MAX, so every k of the official sweep is offered, and all match."""
    P, n_ans, L, H = 60, 70, 2, 2
    T = P + n_ans
    attn, ids = causal_attention(L, H, T, seed=4), ids_for(T, seed=4)
    official = official_pipeline(generate_steps(attn, P, T), torch.cat([ids, torch.tensor([EOS])]))
    feats = extractor().extract(ForwardTrace(attn, (), P, ids), stopped=True,
                                ends_with_terminator=False)
    adapted = adapter_views(feats)
    assert_bitwise(official, adapted)
    assert all(f"lap{k}" in adapted for k in TOP_K_EIGVALS)


def test_f2_contract():
    """Shapes, dtypes, NaN only past the sequence length, finite elsewhere."""
    P, n_ans, L, H = 9, 7, 4, 3
    T = P + n_ans
    attn, ids = causal_attention(L, H, T, seed=5), ids_for(T, seed=5)
    feats = extractor().extract(ForwardTrace(attn, (), P, ids), stopped=True,
                                ends_with_terminator=False)
    for block in (LAP_BLOCK, ATTN_BLOCK):
        x = feats[block]
        assert x.shape == (L, H, K_MAX) and x.dtype == np.float32, (x.shape, x.dtype)
        assert np.isfinite(x[..., :T]).all()
        assert np.isnan(x[..., T:]).all()
    assert feats[LEN_BLOCK].tolist() == [T]


def test_f3_determinism():
    P, n_ans, L, H = 9, 7, 4, 3
    T = P + n_ans
    attn, ids = causal_attention(L, H, T, seed=6), ids_for(T, seed=6)
    trace = ForwardTrace(attn, (), P, ids)
    a = extractor().extract(trace, stopped=True, ends_with_terminator=False)
    b = extractor().extract(trace, stopped=True, ends_with_terminator=False)
    for key in a:
        assert np.array_equal(a[key], b[key], equal_nan=True), key


def test_f3_input_not_mutated():
    """Upstream helpers write into tensors in place in places; ours must stay intact."""
    P, n_ans, L, H = 9, 7, 3, 2
    T = P + n_ans
    attn, ids = causal_attention(L, H, T, seed=7), ids_for(T, seed=7)
    before = [a.clone() for a in attn]
    extractor().extract(ForwardTrace(attn, (), P, ids), stopped=True, ends_with_terminator=False)
    assert all(torch.equal(x, y) for x, y in zip(before, attn))


def test_f6_extractor_sees_no_labels():
    params = set(inspect.signature(OfficialSpectralFeatures.extract).parameters)
    assert params == {"self", "trace", "stopped", "ends_with_terminator"}, params


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


# ----------------------------------------------------------------- deviation measures
# Not pass/fail: they quantify how far our earlier reimplementation sits from the
# official method, for METHOD_CARD.md.

def measure_old_extractor_gap() -> dict:
    from halluc.features.spectral import LapEigvals

    P, n_ans, L, H = 9, 30, 4, 3
    T = P + n_ans
    attn, ids = causal_attention(L, H, T, seed=8), ids_for(T, seed=8)
    trace = ForwardTrace(attn, (), P, ids)
    old = LapEigvals(k=10).extract(trace)["lapeigvals"]
    new = extractor().extract(trace, stopped=True, ends_with_terminator=False)[LAP_BLOCK][..., :10]
    return {"max_abs_diff_k10": float(np.abs(old - new).max()),
            "max_abs_value": float(np.abs(new).max())}


def measure_old_reader_gap() -> dict:
    from sklearn.metrics import roc_auc_score
    from halluc.detectors import PCALinearDetector

    X_tr, y_tr, X_te, y_te = _toy_probe_data(seed=1)
    X_tr = X_tr * np.linspace(0.1, 10, X_tr.shape[1], dtype=np.float32)
    X_te = X_te * np.linspace(0.1, 10, X_te.shape[1], dtype=np.float32)
    old = PCALinearDetector(name="old", blocks=()).estimator({"n_components": 512, "C": 1.0}, 42)
    p_old = old.fit(X_tr, y_tr).predict_proba(X_te)[:, 1]
    p_new = OfficialProbe(random_seed=42).fit(X_tr, y_tr).predict_proba(X_te)[:, 1]
    return {"auroc_old_reader": float(roc_auc_score(y_te, p_old)),
            "auroc_official_reader": float(roc_auc_score(y_te, p_new)),
            "max_abs_prob_diff": float(np.abs(p_old - p_new).max())}


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
    print("\ndeviation, old extractor vs official:", measure_old_extractor_gap())
    print("deviation, old reader vs official   :", measure_old_reader_gap())
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
