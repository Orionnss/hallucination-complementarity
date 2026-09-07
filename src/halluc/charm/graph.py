"""Attention graphs: one attributed graph per generated answer (paper §3).

For a sequence of T tokens the paper defines G = (V, E, X_V, X_E) with

    V = the tokens
    E = ordered pairs (T_i, T_j), i > j, i.e. "token i attends to token j"
    X_E[(i,j)] = alpha_{i,j} in [0,1]^{L*H}   attention paid by i to j, all layer/heads
    X_V[i]     = (alpha_{i,i} | a_i^l)        self-attention, concat residual activations

and sparsifies X_E at a threshold tau (Equation 1), zeroing entries at or below tau and
dropping edges left with no support in any head. Following the paper's own experimental
form (Appendix D.1.2) all prompt->prompt edges are removed, which is what keeps these
graphs small: only the answer tokens carry outgoing edges.

Storage is the crux. A dense X_E would be [n_E, L*H] floats — at Qwen3-14B's L*H=1600
and CoQA's ~7.4k edges that is 15 MB for a *single* sample, so 4 datasets would not fit
in memory, let alone on disk. But X_E^tau is extremely sparse *per entry*: an attention
row sums to 1, so at tau=0.05 at most 20 of a row's entries can survive, and in practice
far fewer. The graph is therefore held in a CSR-style layout over (edge, channel)
non-zeros, which is ~10x smaller and — more importantly — is exactly the layout
`nn.EmbeddingBag` consumes, so the model never has to densify it either (see model.py).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..features.base import ForwardTrace

#: Default sparsification threshold. The paper sweeps tau over {0.5, 0.1, 0.05, 0.01,
#: 0.001} (Table 4) and settles on 0.05 as the best accuracy/footprint trade-off; it is
#: also the value that maximised their validation AUPR.
DEFAULT_TAU = 0.05

#: Edges are gathered in chunks so the intermediate [chunk, L*H] dense block stays
#: bounded no matter how long the sequence is.
_EDGE_CHUNK = 4096


@dataclass(frozen=True)
class GraphSpec:
    """Shapes every graph in a cache shares. Consumed by the model to size its layers."""

    n_channels: int
    """L*H — the width of an attention feature vector."""

    d_act: int
    """Total width of the concatenated activation layers; 0 when activations are off."""

    act_layers: tuple[int, ...]
    tau: float


@dataclass
class AttentionGraph:
    """One sample's sparsified attention graph, on CPU in compact dtypes.

    Edge (k) runs from `edge_index[0, k]` to `edge_index[1, k]`, meaning the former
    attends to the latter. Its L*H-dimensional feature vector is stored sparsely: the
    non-zero channels are `edge_ch[edge_ptr[k]:edge_ptr[k+1]]` with the matching values
    in `edge_val`.
    """

    n_nodes: int
    #: First response position. Prompt->prompt edges are dropped, so only nodes at or
    #: past this index have outgoing edges, and only these are pooled over.
    resp_start: int
    edge_index: np.ndarray  # [2, E] int32 — row 0 attends to row 1
    edge_ptr: np.ndarray  # [E + 1] int64, offsets into edge_ch / edge_val
    edge_ch: np.ndarray  # [nnz] int16/int32 — flat (layer, head) channel index
    edge_val: np.ndarray  # [nnz] float16 — the surviving attention scores
    #: p_{i,j} of Equation 2: 0 = prompt -> response, 1 = response -> response.
    edge_type: np.ndarray  # [E] int8
    node_attn: np.ndarray  # [n_nodes, L*H] float16 — alpha_{i,i}
    node_act: np.ndarray | None  # [n_nodes, d_act] float16, or None

    @property
    def n_edges(self) -> int:
        return int(self.edge_index.shape[1])

    @property
    def n_resp(self) -> int:
        return self.n_nodes - self.resp_start

    def nbytes(self) -> int:
        arrays = [
            self.edge_index, self.edge_ptr, self.edge_ch, self.edge_val,
            self.edge_type, self.node_attn,
        ]
        if self.node_act is not None:
            arrays.append(self.node_act)
        return sum(int(a.nbytes) for a in arrays)


def activation_layers(n_layers: int, fractions=(0.7,)) -> tuple[int, ...]:
    """Activation layers to read, as fractions of model depth.

    The paper probes layers 24/28/32 of a 32-layer LLaMA-2-7B — i.e. roughly 0.75, 0.875
    and 1.0 of depth — and reports (Table 7) that CHARM is robust to the choice while
    concatenating several helps slightly. Fractions rather than absolute indices for the
    same reason `detectors.SAPLMA_DEPTH_FRACTIONS` uses them: the generators here range
    from 28 to 48 layers, and a hardcoded index would probe a different relative depth
    on each.
    """
    return tuple(sorted({max(1, min(n_layers, round(f * n_layers))) for f in fractions}))


@torch.no_grad()
def build_graph(
    trace: ForwardTrace,
    tau: float = DEFAULT_TAU,
    act_layers: tuple[int, ...] = (),
) -> AttentionGraph:
    """Build one sparsified attention graph from a single forward trace.

    Runs on whatever device the trace is on (the attention tensors are the largest
    objects in play, so moving them to CPU first would be the slow way round) and
    returns CPU arrays ready to be cached.
    """
    n_layers, n_heads = trace.n_layers, trace.n_heads
    n_channels = n_layers * n_heads
    T = trace.seq_len
    device = trace.attentions[0].device

    # Match `features/icr.py`: the token *before* the first answer token is the position
    # at which the model commits to the first answer token, so it is treated as part of
    # the response rather than the prompt. Using prompt_len itself would drop the one
    # position every last-token probe in this repo reads.
    resp_start = max(trace.prompt_len - 1, 0)
    n_resp = T - resp_start

    # [C, R, T] — attention paid by each response token, every layer/head. Filled per
    # layer rather than concatenated so the peak allocation is one layer, not two copies
    # of the whole attention stack. float16 halves it again; tau is far above fp16's
    # resolution in this range.
    rows = torch.empty((n_channels, n_resp, T), dtype=torch.float16, device=device)
    diag = torch.empty((n_channels, T), dtype=torch.float16, device=device)
    for layer, attn in enumerate(trace.attentions):
        block = slice(layer * n_heads, (layer + 1) * n_heads)
        rows[block] = attn[:, resp_start:, :].to(torch.float16)
        diag[block] = torch.diagonal(attn, dim1=-2, dim2=-1).to(torch.float16)

    # Strictly lower-triangular in global coordinates: response token at global index
    # resp_start + r may only send to j < resp_start + r. This also removes the
    # self-loops, which are carried as node features instead.
    j_idx = torch.arange(T, device=device)
    i_idx = torch.arange(resp_start, T, device=device).unsqueeze(1)
    rows = torch.where((j_idx.unsqueeze(0) < i_idx).unsqueeze(0), rows, 0.0)
    # Equation 1: zero everything at or below tau, then keep the edges that still have
    # support somewhere. `> tau` and not `>=`, per the paper's case split.
    rows = torch.where(rows > tau, rows, 0.0)

    present = (rows != 0).any(dim=0)  # [R, T]
    edge_r, edge_j = present.nonzero(as_tuple=True)
    n_edges = int(edge_r.numel())

    if n_edges == 0:
        # Degenerate but not impossible: a one-token answer whose attention is spread so
        # thin that nothing clears tau. Returned as an isolated-node graph rather than
        # raising, so a single odd item cannot abort a multi-hour extraction; the model
        # falls back to node features alone for it.
        empty_i32 = np.zeros((2, 0), dtype=np.int32)
        return AttentionGraph(
            n_nodes=T,
            resp_start=resp_start,
            edge_index=empty_i32,
            edge_ptr=np.zeros(1, dtype=np.int64),
            edge_ch=np.zeros(0, dtype=np.int16),
            edge_val=np.zeros(0, dtype=np.float16),
            edge_type=np.zeros(0, dtype=np.int8),
            node_attn=diag.T.float().cpu().numpy().astype(np.float16),
            node_act=_node_activations(trace, act_layers),
        )

    # Gather the surviving (edge, channel) pairs in chunks: rows[:, r, j] for a chunk of
    # edges is [C, chunk], which is the only dense block this function ever holds beyond
    # the attention itself.
    counts, channels, values = [], [], []
    for start in range(0, n_edges, _EDGE_CHUNK):
        stop = min(start + _EDGE_CHUNK, n_edges)
        block = rows[:, edge_r[start:stop], edge_j[start:stop]].T  # [chunk, C]
        # nonzero() on a [chunk, C] tensor returns pairs ordered by row, which is
        # exactly the CSR ordering the offsets below assume.
        local_e, local_c = block.nonzero(as_tuple=True)
        counts.append(torch.bincount(local_e, minlength=stop - start))
        channels.append(local_c)
        values.append(block[local_e, local_c])

    per_edge = torch.cat(counts)
    edge_ptr = torch.zeros(n_edges + 1, dtype=torch.int64, device=device)
    torch.cumsum(per_edge, dim=0, out=edge_ptr[1:])

    ch_dtype = np.int16 if n_channels <= np.iinfo(np.int16).max else np.int32
    global_i = (edge_r + resp_start).to(torch.int32)
    edge_index = torch.stack([global_i, edge_j.to(torch.int32)])

    return AttentionGraph(
        n_nodes=T,
        resp_start=resp_start,
        edge_index=edge_index.cpu().numpy(),
        edge_ptr=edge_ptr.cpu().numpy(),
        edge_ch=torch.cat(channels).cpu().numpy().astype(ch_dtype),
        edge_val=torch.cat(values).cpu().numpy().astype(np.float16),
        # p_{i,j}: the source j is either still in the prompt or already in the response.
        edge_type=(edge_j >= resp_start).to(torch.int8).cpu().numpy(),
        node_attn=diag.T.float().cpu().numpy().astype(np.float16),
        node_act=_node_activations(trace, act_layers),
    )


def _node_activations(trace: ForwardTrace, act_layers: tuple[int, ...]) -> np.ndarray | None:
    """Residual-stream activations for every token, concatenated over `act_layers`.

    `hidden_states` has n_layers + 1 entries (the embedding output is index 0), matching
    how `features/hidden.py` indexes it, so layer indices here mean the same thing as
    SAPLMA's probe layer.
    """
    if not act_layers:
        return None
    top = len(trace.hidden_states) - 1
    blocks = [trace.hidden_states[min(layer, top)].to("cpu", torch.float32) for layer in act_layers]
    return torch.cat(blocks, dim=-1).numpy().astype(np.float16)
