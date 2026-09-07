"""The CHARM network: f = f_pred . f_pool . f_mp (paper §4.1, Appendix D.1.2).

The experimental form the paper actually runs is its Equation 11:

    h_i^(t+1) = up_t( [ h_i^(t) |
                        (1/deg_in(i)) * sum_j msg_t([ h_j^(t) | x_E,(i,j)^tau | p_ij ]) ] )

with up_t and msg_t as MLPs over the concatenation of their arguments, mean aggregation,
and a final mean pool over tokens followed by a dense prediction head.

Two implementation notes matter for this being tractable here:

* **msg_t's first layer is applied to the sparse edge features directly.** Because the
  first layer of an MLP over a concatenation is just the sum of per-argument linear
  maps, `W_e @ x_E` can be computed without ever densifying x_E: it is exactly what
  `nn.EmbeddingBag(mode="sum")` with per-sample weights computes over the CSR layout
  graph.py produces. Densifying instead would cost [E, L*H] floats per batch — around
  4 GB for a heavy CoQA batch at Qwen3-14B's L*H=1600.
* **Only response tokens are pooled.** With prompt->prompt edges removed the prompt
  tokens have no incoming messages and would contribute an unchanged h^(0) to the mean,
  diluting the response signal with a passage-length-dependent constant.

Implemented in plain PyTorch with `index_add_` rather than pulling in PyTorch Geometric:
the single message-passing form above is a few lines of scatter, and the dependency
would otherwise have to be pinned against this project's torch build.
"""

from __future__ import annotations

import torch
from torch import nn

from .data import GraphBatch


class MessagePassing(nn.Module):
    """One layer of Equation 11."""

    def __init__(self, hidden: int, n_channels: int, dropout: float, batch_norm: bool) -> None:
        super().__init__()
        # msg_t's first layer, split by argument. Bias lives on `src` alone so the sum
        # has exactly one, as a single Linear over the concatenation would.
        self.msg_src = nn.Linear(hidden, hidden)
        self.msg_edge = nn.EmbeddingBag(n_channels, hidden, mode="sum", include_last_offset=True)
        self.msg_type = nn.Embedding(2, hidden)  # p_ij: prompt->response / response->response
        self.msg_out = nn.Sequential(nn.ReLU(), nn.Linear(hidden, hidden))
        # up_t over [h_i | aggregated message].
        self.update = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden)
        )
        self.norm = nn.BatchNorm1d(hidden) if batch_norm else nn.Identity()
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, batch: GraphBatch) -> torch.Tensor:
        target, source = batch.edge_index[0], batch.edge_index[1]

        if batch.edge_index.numel():
            edge_embedding = self.msg_edge(
                batch.edge_ch, batch.edge_ptr, per_sample_weights=batch.edge_val
            )
            messages = self.msg_out(
                self.msg_src(h[source]) + edge_embedding + self.msg_type(batch.edge_type)
            )
            aggregated = torch.zeros_like(h)
            aggregated.index_add_(0, target, messages)
            # deg_in(i) = how many tokens i attends to. Clamped because prompt tokens
            # have none; their aggregate is an exact zero either way.
            degree = torch.zeros(h.shape[0], device=h.device, dtype=h.dtype)
            degree.index_add_(0, target, torch.ones_like(target, dtype=h.dtype))
            aggregated = aggregated / degree.clamp(min=1.0).unsqueeze(1)
        else:
            aggregated = torch.zeros_like(h)

        out = self.update(torch.cat([h, aggregated], dim=-1))
        return self.dropout(torch.relu(self.norm(out)))


class CharmNet(nn.Module):
    """f_pred . f_pool . f_mp, for response-level (graph-wise) detection."""

    def __init__(
        self,
        n_channels: int,
        d_act: int,
        hidden: int = 64,
        n_layers: int = 2,
        dropout: float = 0.25,
        batch_norm: bool = True,
        residual: bool = True,
    ) -> None:
        super().__init__()
        self.residual = residual
        self.use_act = d_act > 0

        # h^(0) = x_V,i. Reflexive attention is always present; activations, when used,
        # go through their own encoder first (Appendix D.1.2) and are concatenated to it
        # before message passing. That encoder is also the module the paper regularises
        # with a separate weight decay, so it is kept addressable as `.act_encoder`.
        self.attn_encoder = nn.Linear(n_channels, hidden)
        if self.use_act:
            self.act_encoder = nn.Sequential(
                nn.Linear(d_act, hidden), nn.ReLU(), nn.Linear(hidden, hidden)
            )
        self.input_norm = nn.BatchNorm1d(hidden) if batch_norm else nn.Identity()
        self.node_mix = nn.Linear(2 * hidden if self.use_act else hidden, hidden)

        self.layers = nn.ModuleList(
            MessagePassing(hidden, n_channels, dropout, batch_norm) for _ in range(n_layers)
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1)
        )

    def forward(self, batch: GraphBatch) -> torch.Tensor:
        """Returns one logit per graph."""
        parts = [self.attn_encoder(batch.node_attn)]
        if self.use_act:
            if batch.node_act is None:
                raise ValueError("model was built with activations but the batch carries none")
            parts.append(self.act_encoder(batch.node_act))
        h = self.input_norm(self.node_mix(torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0]))

        for layer in self.layers:
            updated = layer(h, batch)
            h = h + updated if self.residual else updated

        # f_pool: mean over response tokens of each graph.
        pooled = torch.zeros(batch.n_graphs, h.shape[1], device=h.device, dtype=h.dtype)
        pooled.index_add_(0, batch.resp_batch, h[batch.resp_nodes])
        counts = torch.zeros(batch.n_graphs, device=h.device, dtype=h.dtype)
        counts.index_add_(0, batch.resp_batch, torch.ones_like(batch.resp_batch, dtype=h.dtype))
        pooled = pooled / counts.clamp(min=1.0).unsqueeze(1)

        return self.head(pooled).squeeze(-1)

    def parameter_groups(self, weight_decay: float, act_weight_decay: float) -> list[dict]:
        """AdamW groups, with the activation encoder regularised separately.

        Appendix C.4: "we additionally searched over a separate weight decay parameter,
        applied only to the encoder of the activations". The activation block is orders
        of magnitude wider than the attention block (d=5120 vs L*H=1600 on Qwen3-14B),
        so without its own knob it either dominates or has to drag the whole network's
        regularisation with it.
        """
        if not self.use_act:
            return [{"params": list(self.parameters()), "weight_decay": weight_decay}]
        act_params = set(map(id, self.act_encoder.parameters()))
        return [
            {
                "params": [p for p in self.parameters() if id(p) not in act_params],
                "weight_decay": weight_decay,
            },
            {"params": list(self.act_encoder.parameters()), "weight_decay": act_weight_decay},
        ]
