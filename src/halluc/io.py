"""JSON metadata, feature shards, and resumable checkpoints.

Every stage writes a JSON file describing what it did and an append-only checkpoint of
completed item ids so an interrupted run resumes instead of restarting. Float features
live in .npz shards referenced from the JSON — a 16k-dim vector per sample is not
JSON-shaped, but its provenance is.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def _default(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"not JSON serializable: {type(obj)}")


def write_json(path: str | Path, payload: Any) -> Path:
    """Atomic JSON write — a crash mid-write leaves the previous file intact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as fh:
        json.dump(payload, fh, indent=2, default=_default)
    os.replace(tmp, path)
    return path


def read_json(path: str | Path) -> Any:
    with open(path) as fh:
        return json.load(fh)


def git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def provenance() -> dict[str, Any]:
    """Environment fingerprint stamped into every stage's JSON."""
    import torch
    import transformers

    return {
        "git_sha": git_sha(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda": torch.version.cuda,
        "gpu_names": [
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        ],
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


class Checkpoint:
    """Append-only log of completed item ids.

    Written with an fsync per line so a hard kill loses at most the in-flight item.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # The full record is kept per line, not just the id, so this file alone can
        # rebuild the manifest. Otherwise an interrupted run would skip an item on
        # resume (it is checkpointed) while losing its metadata (never written out).
        self._records: dict[str, dict[str, Any]] = {}
        if self.path.exists():
            with open(self.path) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        self._records[record["id"]] = record
                    except (json.JSONDecodeError, KeyError):
                        continue  # truncated final line from a hard kill
        self._fh = open(self.path, "a")

    def __contains__(self, item_id: str) -> bool:
        return item_id in self._records

    def pending(self, item_ids: Iterable[str]) -> list[str]:
        return [i for i in item_ids if i not in self._records]

    def mark(self, item_id: str, **meta: Any) -> None:
        record = {"id": item_id, **meta}
        self._fh.write(json.dumps(record, default=_default) + "\n")
        self._fh.flush()
        os.fsync(self._fh.fileno())
        self._records[item_id] = record

    def records(self) -> dict[str, dict[str, Any]]:
        """Completed records keyed by item id, as recovered from disk.

        `mark` stores the id under "id" (its own positional parameter absorbs any
        "item_id" passed through **meta), so recovered records are normalised here to
        carry both spellings. Without this a resumed run writes manifest entries lacking
        "item_id", which every downstream stage keys on.
        """
        return {
            item_id: {"item_id": item_id, **record}
            for item_id, record in self._records.items()
        }

    def __len__(self) -> int:
        return len(self._records)

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> "Checkpoint":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


class ShardWriter:
    """Buffers per-item feature dicts and flushes them to .npz shards.

    Sharding keeps any single file small enough to load lazily, and lets a resumed run
    append new shards without rewriting old ones.
    """

    def __init__(self, out_dir: str | Path, shard_size: int = 250, prefix: str = "features") -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.shard_size = shard_size
        self.prefix = prefix
        self._buf: dict[str, dict[str, np.ndarray]] = {}
        self._index = self._next_shard_index()

    def _next_shard_index(self) -> int:
        existing = sorted(self.out_dir.glob(f"{self.prefix}_*.npz"))
        if not existing:
            return 0
        return max(int(p.stem.split("_")[-1]) for p in existing) + 1

    def add(self, item_id: str, features: dict[str, np.ndarray]) -> str | None:
        """Buffer one item's features. Returns the shard name if this triggered a flush."""
        self._buf[item_id] = features
        if len(self._buf) >= self.shard_size:
            return self.flush()
        return None

    def pending_shard_name(self) -> str:
        return f"{self.prefix}_{self._index:05d}.npz"

    def flush(self) -> str | None:
        if not self._buf:
            return None
        name = self.pending_shard_name()
        # Flatten to "<item_id>/<feature_name>" so one npz holds many items.
        flat = {
            f"{item_id}/{fname}": arr
            for item_id, feats in self._buf.items()
            for fname, arr in feats.items()
        }
        tmp = self.out_dir / (name + ".tmp")
        # Write through a file handle: given a path not ending in .npz, numpy appends
        # the suffix itself and the rename would target a file that does not exist.
        with open(tmp, "wb") as fh:
            np.savez_compressed(fh, **flat)
        os.replace(tmp, self.out_dir / name)
        self._buf.clear()
        self._index += 1
        return name

    def __enter__(self) -> "ShardWriter":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.flush()


def load_features(
    shard_dir: str | Path, feature_name: str, item_ids: list[str]
) -> np.ndarray:
    """Stack one feature block across items, in the order given by `item_ids`."""
    shard_dir = Path(shard_dir)
    found: dict[str, np.ndarray] = {}
    wanted = set(item_ids)
    for shard in sorted(shard_dir.glob("*.npz")):
        with np.load(shard) as data:
            for key in data.files:
                item_id, _, fname = key.rpartition("/")
                if fname == feature_name and item_id in wanted:
                    found[item_id] = data[key]
    missing = wanted - found.keys()
    if missing:
        raise KeyError(
            f"{len(missing)} items missing feature {feature_name!r} in {shard_dir} "
            f"(e.g. {sorted(missing)[:3]})"
        )
    return np.stack([found[i] for i in item_ids])
