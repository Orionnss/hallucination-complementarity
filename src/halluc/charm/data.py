"""Collating attention graphs into batches, and the in-RAM graph cache.

Graphs are batched the standard GNN way — disjoint union into one big graph, with node
indices offset per member and a `batch` vector recording which graph each node came
from. Pooling then becomes a scatter-mean over that vector.

The cache exists because the CV protocol demands it. Stage 3 fits every detector 5 seeds
x 5 outer folds x |grid| times over the same samples; re-running the LLM for each of
those would cost days. So the generator runs once per item, and the resulting graphs
stay in memory for every subsequent fit. Nothing is written to disk.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .graph import AttentionGraph, GraphSpec


@dataclass
class GraphBatch:
    """A disjoint union of graphs, on the training device."""

    node_attn: torch.Tensor  # [N, C] float32
    node_act: torch.Tensor | None  # [N, d_act] float32
    edge_index: torch.Tensor  # [2, E] int64, already node-offset
    edge_ch: torch.Tensor  # [nnz] int64
    edge_val: torch.Tensor  # [nnz] float32
    edge_ptr: torch.Tensor  # [E + 1] int64
    edge_type: torch.Tensor  # [E] int64
    #: Graph membership for response nodes only — prompt nodes are never pooled, since
    #: with prompt->prompt edges removed they are message sources that never update.
    resp_nodes: torch.Tensor  # [n_resp_total] int64
    resp_batch: torch.Tensor  # [n_resp_total] int64
    n_nodes: int
    n_graphs: int


def collate(graphs: list[AttentionGraph], device: torch.device) -> GraphBatch:
    """Disjoint-union `graphs` into one batch on `device`."""
    node_offset, edge_offset = 0, 0
    attn, acts = [], []
    idx, ch, val, ptr, etype = [], [], [], [], []
    resp_nodes, resp_batch = [], []

    for graph_id, graph in enumerate(graphs):
        attn.append(graph.node_attn)
        if graph.node_act is not None:
            acts.append(graph.node_act)
        idx.append(graph.edge_index.astype(np.int64) + node_offset)
        ch.append(graph.edge_ch.astype(np.int64))
        val.append(graph.edge_val.astype(np.float32))
        # edge_ptr[0] is always 0; drop it and shift so the concatenation stays a valid
        # single offsets vector for the whole batch.
        ptr.append(graph.edge_ptr[1:].astype(np.int64) + edge_offset)
        etype.append(graph.edge_type.astype(np.int64))
        span = np.arange(graph.resp_start, graph.n_nodes, dtype=np.int64) + node_offset
        resp_nodes.append(span)
        resp_batch.append(np.full(span.shape, graph_id, dtype=np.int64))
        node_offset += graph.n_nodes
        edge_offset += int(graph.edge_ptr[-1])

    as_tensor = lambda parts, dtype: torch.from_numpy(  # noqa: E731
        np.concatenate(parts)
    ).to(device=device, dtype=dtype, non_blocking=True)

    return GraphBatch(
        node_attn=torch.from_numpy(np.concatenate(attn)).to(device, torch.float32),
        node_act=(
            torch.from_numpy(np.concatenate(acts)).to(device, torch.float32) if acts else None
        ),
        edge_index=torch.from_numpy(np.concatenate(idx, axis=1)).to(device, torch.int64),
        edge_ch=as_tensor(ch, torch.int64),
        edge_val=as_tensor(val, torch.float32),
        edge_ptr=torch.cat(
            [torch.zeros(1, dtype=torch.int64, device=device), as_tensor(ptr, torch.int64)]
        ),
        edge_type=as_tensor(etype, torch.int64),
        resp_nodes=as_tensor(resp_nodes, torch.int64),
        resp_batch=as_tensor(resp_batch, torch.int64),
        n_nodes=node_offset,
        n_graphs=len(graphs),
    )


class GraphCache:
    """Every sample's graph, keyed by item id, held in RAM for the whole run."""

    def __init__(self, spec: GraphSpec) -> None:
        self.spec = spec
        self._graphs: dict[str, AttentionGraph] = {}

    def add(self, item_id: str, graph: AttentionGraph) -> None:
        self._graphs[item_id] = graph

    def __contains__(self, item_id: str) -> bool:
        return item_id in self._graphs

    def __len__(self) -> int:
        return len(self._graphs)

    def get(self, item_ids: list[str]) -> list[AttentionGraph]:
        return [self._graphs[i] for i in item_ids]

    def nbytes(self) -> int:
        return sum(g.nbytes() for g in self._graphs.values())

    def stats(self) -> dict:
        if not self._graphs:
            return {"n_graphs": 0}
        edges = np.array([g.n_edges for g in self._graphs.values()])
        nnz = np.array([len(g.edge_val) for g in self._graphs.values()])
        nodes = np.array([g.n_nodes for g in self._graphs.values()])
        return {
            "n_graphs": len(self._graphs),
            "mean_nodes": round(float(nodes.mean()), 1),
            "mean_edges": round(float(edges.mean()), 1),
            "p95_edges": int(np.percentile(edges, 95)),
            "max_edges": int(edges.max()),
            "mean_edge_nnz": round(float(nnz.mean()), 1),
            # How much of a dense [n_E, L*H] X_E is actually stored. This is the number
            # that decides whether the cache fits in memory at all.
            "edge_feature_density": round(
                float(nnz.sum() / max(edges.sum() * self.spec.n_channels, 1)), 5
            ),
            "resident_gib": round(self.nbytes() / 1024**3, 2),
        }


def batch_indices(
    graphs: list[AttentionGraph],
    order: np.ndarray,
    batch_size: int,
    max_edges: int,
) -> list[np.ndarray]:
    """Split `order` into batches of `batch_size`, subdividing edge-heavy ones.

    A fixed sample count is what the paper specifies (batch size 32), but CoQA's long
    passages make graph size vary by more than an order of magnitude, and a batch that
    happens to collect several of them would blow up the dense [E, hidden] message
    tensor. Batches over the edge budget are therefore split further — the effective
    batch is never *larger* than the paper's, only smaller on the heavy tail.
    """
    batches, current, current_edges = [], [], 0
    for pos in order:
        n_edges = graphs[pos].n_edges
        over_budget = current and current_edges + n_edges > max_edges
        if len(current) == batch_size or over_budget:
            batches.append(np.array(current, dtype=np.int64))
            current, current_edges = [], 0
        current.append(pos)
        current_edges += n_edges
    if current:
        batches.append(np.array(current, dtype=np.int64))
    return batches
