"""Per-fit checkpoints, so a crash costs one network fit rather than the whole run.

Stage 6's work decomposes into (seed, fold, grid point) fits — 100 of them at the
default grid, several hours in total. Each one is independent given the fold split, so
each is checkpointed as it finishes: its validation and test scores, the record stage 6
reports, and the trained weights. A resumed run replays the completed fits from disk and
only trains what is missing.

What is *not* checkpointed is the graph cache. That is the point of the design — the
graphs never touch disk — so a resumed run rebuilds them by re-running the generator's
forward pass, roughly 25 minutes for all four datasets. Only the expensive half is
recovered.

The correctness risk in any resume is silently mixing results computed over different
data. Two guards:

* a **fingerprint** over the seed's item ids, fold count and scope, recorded once per
  seed directory; a run whose draw does not match refuses to reuse the directory rather
  than blending two sample sets;
* each fit file carries **its own hyperparameters**, and is keyed by their hash, so
  editing or reordering the grid re-trains the affected points instead of loading
  someone else's scores under a reused index.

Fold-level results — which grid point won, the decision threshold — are cheap and
deterministic, so they are recomputed from the cached fits rather than stored.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from ..io import read_json, write_json


def seed_fingerprint(item_ids: list[str], n_folds: int, scope: str, seed: int) -> str:
    """Identity of the sample set a seed's fits were computed over.

    The item ids are hashed in order, so both a different draw and a different ordering
    invalidate the checkpoint — the cached score arrays are positional.
    """
    digest = hashlib.sha256()
    digest.update(f"{scope}|{seed}|{n_folds}|{len(item_ids)}|".encode())
    for item_id in item_ids:
        digest.update(item_id.encode())
        digest.update(b"\0")
    return digest.hexdigest()[:16]


def _params_key(params) -> str:
    """Short stable hash of one grid point, used in the fit's filename."""
    payload = json.dumps(asdict(params), sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:10]


class FitStore:
    """Completed fits for one (scope, seed), on disk under the seed's directory."""

    def __init__(self, root: Path, fingerprint: str, save_models: bool = True) -> None:
        self.root = Path(root)
        self.fingerprint = fingerprint
        self.save_models = save_models
        self.root.mkdir(parents=True, exist_ok=True)

        manifest = self.root / "manifest.json"
        if manifest.exists():
            recorded = read_json(manifest).get("fingerprint")
            if recorded != fingerprint:
                raise SystemExit(
                    f"{self.root} holds fits for a different sample draw "
                    f"(fingerprint {recorded}, this run {fingerprint}). The draw depends "
                    f"on the labels, n_per_seed and the dataset list, so one of those has "
                    f"changed. Delete that directory to re-train this seed."
                )
        else:
            write_json(manifest, {"fingerprint": fingerprint})

    def _paths(self, fold: int, params) -> tuple[Path, Path]:
        stem = f"fit_f{fold}_{_params_key(params)}"
        return self.root / f"{stem}.npz", self.root / f"{stem}.pt"

    def load(self, fold: int, params) -> dict | None:
        """A completed fit, or None if it has not been trained yet."""
        scores_path, _ = self._paths(fold, params)
        if not scores_path.exists():
            return None
        try:
            with np.load(scores_path, allow_pickle=False) as data:
                record = json.loads(str(data["record"].item()))
                stored = json.loads(str(data["params"].item()))
                if stored != asdict(params):
                    return None  # hash collision, or a params schema change
                return {
                    "val_scores": data["val_scores"],
                    "eval_scores": data["eval_scores"],
                    "record": record,
                }
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            # A fit interrupted mid-write. Retrain it rather than trust a partial file.
            return None

    def save(self, fold: int, params, val_scores, eval_scores, record, state) -> None:
        scores_path, model_path = self._paths(fold, params)
        tmp = scores_path.with_suffix(".npz.tmp")
        with open(tmp, "wb") as fh:
            np.savez_compressed(
                fh,
                val_scores=val_scores,
                eval_scores=eval_scores,
                record=np.array(json.dumps(record)),
                params=np.array(json.dumps(asdict(params))),
            )
        os.replace(tmp, scores_path)

        if self.save_models and state is not None:
            tmp_model = model_path.with_suffix(".pt.tmp")
            torch.save({"params": asdict(params), "state_dict": state}, tmp_model)
            os.replace(tmp_model, model_path)

    def n_completed(self) -> int:
        return len(list(self.root.glob("fit_f*.npz")))
