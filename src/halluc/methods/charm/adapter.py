"""Stage-1 adapter: the shared trace in, the official CHARM graph and activations out.

data_prep_2/data_collection.py does this per item on `model.generate(output_attentions=True,
output_hidden_states=True, return_dict_in_generate=True)`:

  data = get_data_object(model_output.attentions, threshold=0.05, prompt_graph=False)
  data.x, data.edge_attr -> float32 -> float16
  acts, layers = get_activations(model_output.hidden_states, act_layers)
  acts -> float32 -> float16

The native view of the shared trace is that generate() output, cut to the kept answer by
the harness. It is moved to the CPU first: `get_attention_matrix` calls `.cpu()` on every
(step, layer, head) slice, which is ~50k device copies per item on a GPU and a no-op on
CPU tensors. The values are the same either way.

Stored per item (ragged): the graph's node features x [N, L*H], edge_index [2, E],
edge_attr [E, L*H] (float16, as upstream stores them), the activations of one layer
[N, d] (float16), and upstream's prompt_len (= P - 1: the last prompt token predicts the
first answer token). head/layer/response_index are rebuilt from these at training time.
"""

from __future__ import annotations

import contextlib
import io

import numpy as np
import torch

from ...features.base import FEATURES, FeatureExtractor, ForwardTrace
from .upstream_spec import ACT_DEPTH_FRACTION, CONFIG, GRAPH_KWARGS, SPEC, act_layer, load

BLOCKS = ("charm_x", "charm_edge_index", "charm_edge_attr", "charm_act",
          "charm_prompt_len", "charm_act_layer", "charm_n_heads")


@FEATURES.register("charm_official")
class OfficialCharmGraph(FeatureExtractor):
    name = "charm_official"
    needs = frozenset({"attentions", "hidden_states"})
    store = "methods"

    def __init__(self) -> None:
        self.up = load()

    def params(self) -> dict:
        return {"graph": GRAPH_KWARGS, "act_depth_fraction": ACT_DEPTH_FRACTION,
                "config": CONFIG, "upstream": SPEC.url, "commit": SPEC.commit}

    def extract(self, trace: ForwardTrace) -> dict[str, np.ndarray]:
        if trace.step_attentions is None or trace.step_hidden_states is None:
            raise ValueError("charm_official needs the native generate() view with "
                             "attentions and hidden states (ADDING_A_METHOD.md §5.0)")
        attentions = tuple(tuple(a.cpu() for a in step) for step in trace.step_attentions)
        hidden = tuple(tuple(h.cpu() for h in step) for step in trace.step_hidden_states)
        n_layers = len(attentions[0])
        layer = act_layer(n_layers)

        with contextlib.redirect_stdout(io.StringIO()):   # upstream prints per graph
            data = self.up["graphs"].get_data_object(attentions, **GRAPH_KWARGS)
        x = data.x.to(torch.float32).to(torch.float16)
        edge_attr = data.edge_attr.to(torch.float32).to(torch.float16)
        acts, _ = self.up["acts"].get_activations(hidden, [layer])
        acts = acts.to(torch.float32).to(torch.float16)

        if data.edge_index.numel() and int(data.edge_index.max()) >= 2**31:
            raise ValueError("edge index does not fit int32")
        return {
            "charm_x": x.numpy(),
            "charm_edge_index": data.edge_index.to(torch.int32).numpy(),
            "charm_edge_attr": edge_attr.numpy(),
            "charm_act": acts.numpy(),
            "charm_prompt_len": np.array([int(data.prompt_len)], dtype=np.int32),
            "charm_act_layer": np.array([layer], dtype=np.int32),
            # Needed to rebuild Data.head / Data.layer (x is [N, L*H], layer-major).
            "charm_n_heads": np.array([attentions[0][0].shape[1]], dtype=np.int32),
        }
