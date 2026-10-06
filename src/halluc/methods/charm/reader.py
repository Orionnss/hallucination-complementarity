"""Native reader and grid registration for the official CHARM graphs.

run_exp.py loads a YAML config into a Namespace and calls training.main, which:
  get_transforms(args)      -> pre-transforms (LabelPooling, ThresholdAttention(0.05),
                               MarkPromptEdges for this config), applied once per graph
  get_data(...)             -> AttentionDataset + `hydrate` with the activations
  get_model(args, ...)      -> CHARM
  train_model(...)          -> BCEWithLogits, Adam, plateau scheduler on the validation
                               AUPR of the hallucination class, early stop (patience),
                               checkpoint of the best validation epoch

This reader runs the same functions on the same objects. What it does itself, and why:

  * builds each `Data` graph from the stored arrays (adapter.py) instead of loading
    AttentionDataset from a dataset directory; `hydrate` sets `data.act` to float16 the
    same way. head/layer/response_index are rebuilt as get_data_object makes them.
  * takes its validation set (20%, stratified) from its own training rows. CHARM's own
    splits are 60/20/20 or carve validation out of the provided training set.
  * labels: CHARM's y is 1 for a CORRECT answer (data_prep_2 annotation = correctness;
    evaluate() scores hallucination as AP(1 - y, -logit)). The reader trains on 1 - y
    and returns 1 - sigmoid(logit) as the hallucination probability.
  * train_model also evaluates a "test" loader each epoch, for logging only; it gets the
    validation loader, because the reader must never see test labels.
  * loaders run in the main process (num_workers=0): data loading only.
"""

from __future__ import annotations

import contextlib
import io
import tempfile
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split

from ...config import Config
from ...grid import BLOCKS, BlockSpec, Reader, READERS
from .adapter import BLOCKS as ARRAYS
from .upstream_spec import CONFIG, SPEC, load


def official_args(seed: int) -> Namespace:
    """The official YAML as run_exp.py loads it, plus the seed."""
    with open(SPEC.path / CONFIG) as fh:
        cfg = yaml.safe_load(fh)
    cfg["seed"] = seed
    cfg["log"] = False
    cfg["verbose"] = False
    return Namespace(**cfg)


def to_graph(item: dict[str, np.ndarray], n_heads: int, y_correct: float):
    """One stored item -> the `Data` object get_data_object + hydrate would give."""
    from torch_geometric.data import Data

    x = torch.from_numpy(item["charm_x"])
    n_nodes, n_feats = x.shape
    n_layers = n_feats // n_heads
    prompt_len = int(item["charm_prompt_len"][0])
    return Data(
        x=x,
        edge_index=torch.from_numpy(item["charm_edge_index"]).to(torch.int64),
        edge_attr=torch.from_numpy(item["charm_edge_attr"]),
        head=torch.arange(n_heads).repeat(n_layers),
        layer=torch.arange(n_layers).repeat_interleave(n_heads),
        response_index=torch.arange(prompt_len, n_nodes),
        prompt_len=prompt_len,
        act=torch.from_numpy(item["charm_act"]).to(torch.float16),
        y=torch.tensor(float(y_correct)),
    )


class OfficialCharm(BaseEstimator, ClassifierMixin):
    def __init__(self, n_heads: int, random_seed: int = 0) -> None:
        self.n_heads = n_heads
        self.random_seed = random_seed

    def _graphs(self, items, y_correct, transform):
        return [transform(to_graph(it, self.n_heads, yc)) for it, yc in zip(items, y_correct)]

    def fit(self, X, y):
        from torch_geometric import seed_everything
        from torch_geometric.loader import DataLoader

        up = load()
        training = up["training"]
        y = np.asarray(y, dtype=int)
        self.classes_ = np.array([0, 1])
        self.args_ = args = official_args(self.random_seed)
        seed_everything(args.seed)
        with contextlib.redirect_stdout(io.StringIO()):
            T, _, _, _ = training.get_transforms(args)
        self.transform_ = T

        tr, va = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y,
                                  random_state=self.random_seed)
        correct = 1.0 - y
        train_list = self._graphs(X[tr], correct[tr], T)
        val_list = self._graphs(X[va], correct[va], T)
        dims = {"act": train_list[0]["act"].shape[-1]}
        train_loader = DataLoader(train_list, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_list, batch_size=64, shuffle=False)

        with tempfile.TemporaryDirectory() as tmp, \
                contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            model = training.get_model(args, train_list, dims)
            pack = {"checkpoint_path": str(Path(tmp) / "best.pt"), "args": args, "git_sha": None}
            self.result_ = training.train_model(
                model, train_loader, val_loader, val_loader, num_epochs=args.num_epochs,
                lr=args.learning_rate, scheduler=args.scheduler, weight_decay=args.weight_decay,
                weight_decay_target=args.weight_decay_target,
                weight_decay_for_target=args.weight_decay_for_target, balance=args.balance,
                train_prebatcher=None, patience=args.patience, log=False,
                checkpoint_pack=pack, verbose=False)
            best = torch.load(pack["checkpoint_path"], weights_only=False)
        model.load_state_dict(best["model_state_dict"])
        self.model_ = model.eval()
        self.device_ = next(model.parameters()).device
        return self

    @torch.no_grad()
    def predict_proba(self, X):
        from torch_geometric.loader import DataLoader

        training = load()["training"]
        graphs = self._graphs(X, np.zeros(len(X)), self.transform_)
        out = []
        for batch in DataLoader(graphs, batch_size=64, shuffle=False):
            batch = batch.to(self.device_)
            logits = self.model_(*training.prepare_args(batch))
            out.append(logits.reshape(-1).double().cpu())
        p_correct = torch.sigmoid(torch.cat(out)).numpy()
        p_halluc = 1.0 - p_correct
        return np.stack([1.0 - p_halluc, p_halluc], axis=1)


def _n_heads(data) -> int:
    """Heads per layer, as stored by the adapter (one generator per run)."""
    return int(data["charm_official"][0]["charm_n_heads"][0])


def native_threshold(scores: np.ndarray) -> np.ndarray:
    """CHARM reports threshold-free metrics only; a 0.5 cut on the probability."""
    return (scores > 0.5).astype(int)


def _source(cfg: Config, dataset: str):
    return cfg.stage_dir("methods", "charm_official", dataset)


#: The number of heads per layer reaches the reader as a (label-free) view parameter.
READERS.register("charm_official")(lambda: Reader(
    name="charm_official",
    grid=[{}],
    build=lambda p, s: OfficialCharm(n_heads=p["n_heads"], random_seed=s),
    accepts=frozenset({"graph"}),
    native_decision=native_threshold,
    note="official CHARM GNN, movies_att_act config, best epoch by validation AUPR(hallu)",
))


BLOCKS.register("charm_official")(lambda: BlockSpec(
    name="charm_official", ftype="graph", arrays=ARRAYS, source=_source, ragged=True,
    view_grid=lambda data: [{"n_heads": _n_heads(data)}],
))
