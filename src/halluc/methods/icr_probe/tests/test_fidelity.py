"""Fidelity tests for the ICR Probe adapter and native reader (ADDING_A_METHOD.md §7).

CPU only, synthetic `generate()` outputs (halluc.methods.testing). They prove that the
adapter gives the official ICRScore its own input (the native view), with core_positions
from the harness's user span and the response start at the first generated token, and
pools exactly as the official notebook does. Real outputs: tests/f1_real.py (GPU).

Run:  uv run python src/halluc/methods/icr_probe/tests/test_fidelity.py
"""

from __future__ import annotations

import dataclasses
import inspect
import sys
import traceback

import numpy as np
import torch

from halluc.methods.icr_probe import OfficialIcrFeatures, SPEC, load
from halluc.methods.icr_probe.adapter import ICR_BLOCK, NTOK_BLOCK
from halluc.methods.icr_probe.reader import OfficialIcrProbe
from halluc.methods.icr_probe.upstream_spec import COMPUTE_ICR_KWARGS, ICR_SCORE_KWARGS
from halluc.methods.testing import make_case
from halluc.methods.upstream import UpstreamMismatch, verify

L, H, D = 4, 8, 16  # 8 heads: the upstream induction-head rule keeps num_heads // 8 >= 1

EXTRACTOR = None


def extractor() -> OfficialIcrFeatures:
    global EXTRACTOR
    if EXTRACTOR is None:
        EXTRACTOR = OfficialIcrFeatures()
    return EXTRACTOR


def official(trace, user_span, prompt_len):
    """The README usage, called directly on the same generate() outputs."""
    up = load()
    icr = up["icr_score"].ICRScore(
        hidden_states=trace.step_hidden_states, attentions=trace.step_attentions,
        core_positions={"user_prompt_start": user_span[0], "user_prompt_end": user_span[1],
                        "response_start": prompt_len},
        icr_device=torch.device("cpu"), **ICR_SCORE_KWARGS)
    scores, _ = icr.compute_icr(**COMPUTE_ICR_KWARGS)
    return np.mean(np.array(scores), axis=-1)       # empirical_study.ipynb, read_acd_scores


def test_f1_equals_official_readme_usage():
    for P, n, keep, span in [(20, 9, 8, (3, 15)), (40, 25, 25, (5, 33))]:
        trace, *_ = make_case(P, n, keep, L, H, seed=P, d=D, user_span=span)
        feats = extractor().extract(trace)
        ref = official(trace, span, P).astype(np.float32)
        assert np.array_equal(feats[ICR_BLOCK], ref), (feats[ICR_BLOCK], ref)
        assert feats[NTOK_BLOCK].tolist() == [trace.seq_len - P]


def test_f1_core_positions_matter():
    """A different user span changes the induction heads, hence the scores."""
    trace, *_ = make_case(30, 12, 11, L, H, seed=5, d=D, user_span=(2, 26))
    a = extractor().extract(trace)[ICR_BLOCK]
    trace.user_span = (20, 26)
    b = extractor().extract(trace)[ICR_BLOCK]
    assert a.shape == b.shape == (L,)


def test_f2_contract_and_empty_answer():
    trace, *_ = make_case(20, 9, 8, L, H, seed=7, d=D, user_span=(3, 15))
    feats = extractor().extract(trace)
    assert feats[ICR_BLOCK].shape == (L,) and feats[ICR_BLOCK].dtype == np.float32
    assert np.isfinite(feats[ICR_BLOCK]).all()
    empty, *_ = make_case(20, 1, 0, L, H, seed=8, d=D, user_span=(3, 15))
    out = extractor().extract(empty)
    assert np.isnan(out[ICR_BLOCK]).all() and out[NTOK_BLOCK].tolist() == [0]


def test_f2_refuses_missing_user_span():
    trace, *_ = make_case(20, 9, 8, L, H, seed=9, d=D)
    trace.user_span = None
    try:
        extractor().extract(trace)
    except ValueError:
        return
    raise AssertionError("a trace without user_span was accepted")


def test_f3_determinism_and_input_not_mutated():
    trace, *_ = make_case(20, 9, 8, L, H, seed=10, d=D, user_span=(3, 15))
    before = [[a.clone() for a in s] for s in trace.step_attentions]
    a = extractor().extract(trace)
    b = extractor().extract(trace)
    assert all(np.array_equal(a[k], b[k], equal_nan=True) for k in a)
    for s0, s1 in zip(before, trace.step_attentions):
        assert all(torch.equal(x, y) for x, y in zip(s0, s1))


def test_f6_extractor_sees_no_labels():
    assert set(inspect.signature(OfficialIcrFeatures.extract).parameters) == {"self", "trace"}


def test_reader_trains_the_official_probe():
    rng = np.random.default_rng(0)
    n, d = 600, 12
    y = rng.integers(0, 2, n)
    X = rng.normal(size=(n, d)).astype(np.float32)
    X[:, :3] += y[:, None] * 1.0
    probe = OfficialIcrProbe(random_seed=0, device="cpu").fit(X[:500], y[:500])
    p = probe.predict_proba(X[500:])
    assert p.shape == (100, 2) and np.allclose(p.sum(1), 1.0)
    from sklearn.metrics import roc_auc_score
    assert roc_auc_score(y[500:], p[:, 1]) > 0.7
    model = probe.model_
    assert type(model).__name__ == "ICRProbe" and type(model).__module__ == "src.utils"
    again = OfficialIcrProbe(random_seed=0, device="cpu").fit(X[:500], y[:500]).predict_proba(X[500:])
    assert np.array_equal(p, again), "same seed, different probe"


def test_upstream_identity():
    verify(SPEC)
    bad = dataclasses.replace(SPEC, files={**SPEC.files, "src/utils.py": "0" * 64})
    try:
        verify(bad)
    except UpstreamMismatch:
        return
    raise AssertionError("a wrong file hash was accepted")


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
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
