"""Fidelity tests for the CHARM adapter and native reader (ADDING_A_METHOD.md §7).

CPU only, synthetic `generate()` outputs (halluc.methods.testing). They prove that:

  * the adapter's blocks equal what data_prep_2/data_collection.py computes and stores
    from the same generate() output (get_data_object + get_activations, float16), and
  * the reader rebuilds exactly the `Data` object that upstream builds and hydrates, so
    the official transforms, model and training loop see the official input.

Run:  uv run python src/halluc/methods/charm/tests/test_fidelity.py
"""

from __future__ import annotations

import contextlib
import dataclasses
import inspect
import io
import sys
import traceback

import numpy as np
import torch

from halluc.methods.charm import OfficialCharmGraph, SPEC, load
from halluc.methods.charm.reader import OfficialCharm, to_graph
from halluc.methods.charm.upstream_spec import GRAPH_KWARGS, act_layer
from halluc.methods.testing import make_case
from halluc.methods.upstream import UpstreamMismatch, verify

L, H, D = 3, 2, 8

EXTRACTOR = None


def extractor() -> OfficialCharmGraph:
    global EXTRACTOR
    if EXTRACTOR is None:
        EXTRACTOR = OfficialCharmGraph()
    return EXTRACTOR


def upstream_collection(trace):
    """data_collection.py: get_data_object + get_activations on generate() output."""
    up = load()
    with contextlib.redirect_stdout(io.StringIO()):
        data = up["graphs"].get_data_object(trace.step_attentions, **GRAPH_KWARGS)
    data.x = data.x.to(torch.float32).to(torch.float16)
    data.edge_attr = data.edge_attr.to(torch.float32).to(torch.float16)
    acts, _ = up["acts"].get_activations(trace.step_hidden_states, [act_layer(L)])
    return data, acts.to(torch.float32).to(torch.float16)


def test_f1_blocks_equal_upstream_collection():
    for P, n, keep in [(12, 6, 5), (20, 15, 15), (12, 1, 0)]:
        trace, *_ = make_case(P, n, keep, L, H, seed=P + n, d=D)
        feats = extractor().extract(trace)
        data, acts = upstream_collection(trace)
        assert np.array_equal(feats["charm_x"], data.x.numpy())
        assert np.array_equal(feats["charm_edge_index"].astype(np.int64), data.edge_index.numpy())
        assert np.array_equal(feats["charm_edge_attr"], data.edge_attr.numpy())
        assert np.array_equal(feats["charm_act"], acts.numpy())
        assert int(feats["charm_prompt_len"][0]) == data.prompt_len == P - 1
        assert feats["charm_x"].shape == (trace.seq_len, L * H)


def test_f1_reader_rebuilds_upstream_data():
    """to_graph(stored blocks) == upstream Data + hydrate(act), field by field."""
    trace, *_ = make_case(15, 8, 7, L, H, seed=3, d=D)
    feats = extractor().extract(trace)
    data, acts = upstream_collection(trace)
    graph = to_graph(feats, n_heads=H, y_correct=1.0)
    for field in ("x", "edge_index", "edge_attr", "head", "layer", "response_index"):
        assert torch.equal(getattr(graph, field), getattr(data, field)), field
    assert graph.prompt_len == data.prompt_len
    assert torch.equal(graph.act, acts.to(torch.float16))


def test_f3_determinism_and_input_not_mutated():
    trace, *_ = make_case(12, 6, 5, L, H, seed=4, d=D)
    before = [[a.clone() for a in s] for s in trace.step_attentions]
    a = extractor().extract(trace)
    b = extractor().extract(trace)
    assert all(np.array_equal(a[k], b[k]) for k in a)
    for s0, s1 in zip(before, trace.step_attentions):
        assert all(torch.equal(x, y) for x, y in zip(s0, s1))


def test_f6_extractor_sees_no_labels():
    assert set(inspect.signature(OfficialCharmGraph.extract).parameters) == {"self", "trace"}


def test_reader_runs_official_training_deterministically():
    items, y = [], []
    for i in range(80):
        trace = make_case(10, 6, 5, L, H, seed=100 + i, d=D)[0]
        items.append(extractor().extract(trace))
        y.append(i % 2)
    X = np.empty(len(items), dtype=object)
    for k, it in enumerate(items):
        X[k] = it
    y = np.array(y)
    p1 = OfficialCharm(n_heads=H, random_seed=0).fit(X[:60], y[:60]).predict_proba(X[60:])
    p2 = OfficialCharm(n_heads=H, random_seed=0).fit(X[:60], y[:60]).predict_proba(X[60:])
    assert p1.shape == (20, 2) and np.isfinite(p1).all() and np.allclose(p1.sum(1), 1.0)
    assert np.array_equal(p1, p2), "same seed, different CHARM"
    model = OfficialCharm(n_heads=H, random_seed=0).fit(X[:60], y[:60]).model_
    assert type(model).__name__ == "CHARM" and type(model).__module__ == "model.mp"


def test_upstream_identity():
    verify(SPEC)
    bad = dataclasses.replace(SPEC, files={**SPEC.files, "model/mp.py": "0" * 64})
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
