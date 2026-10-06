"""Pinned identity of the official CHARM code, and the modules the adapter runs."""

from __future__ import annotations

from ..upstream import UpstreamSpec, load_upstream

SPEC = UpstreamSpec(
    name="charm",
    url="https://github.com/Noired/charm",
    commit="7ab3ab7f7a49fd6ca23defc44ddcd6c794598c61",
    files={
        # Extraction, as called by data_prep_2/data_collection.py on generate() output.
        "dataset/graphs.py": "46e6f93600913cbf6e3a6cc3e3b92efcd868eb41d661b4a5ba89cbc20a0141a1",
        "dataset/acts.py": "f9aef5a000d62a7ef60ceab7694695fe832ffad230e839cad93c52da35b09e4d",
        # Training, as run by run_exp.py -> training.main.
        "dataset/transforms.py": "8ea80973564d87cbe65ca705ab96af56864b58a27cbe387d7c59b0f20080c2e0",
        "dataset/dataset.py": "ba83c6b71b6a03beee6cae789a9c0099d7105ce81323917ce04509e165d9a127",
        "dataset/prebatch.py": "8a60f39680de635fccd07b84f74a0449d045c9a8fc1d1e739bf5f63259bb2b10",
        "model/mp.py": "1542a9846e6a21b95a997eff684314aa4f9d89d42a7cd2386df2b02660dd7550",
        "model/nn.py": "f05bf22f5fc51f0f253d6a0ae53be9e338c8bc9178c2c4f8055b75ec035578ad",
        "training.py": "116250e764165bf9c439ee64259c6b359d97799326788b56766609328dd3673b",
        "baseline/act_baseline_fitting.py":
            "55c77de77a4abe59e4960cebb095bfa38d9a9f23e2278eebabcb06c77f52c141",
        "baseline/baseline_utils.py":
            "106ed61c71316d93283aa766ed73c941ebe021753ef3c9fb5d396c501675a2ec",
        # The configuration the native reader runs (see CONFIG below).
        "configs/charm/movies_att_act.yaml":
            "bddbc64bf8fe6e628d41c354376179ff417f9fff8f37cfc4aaf007c288c43226",
    },
    # `dataset`, `model` have no __init__; `baseline/__init__.py` is not needed.
    packages=("dataset", "model", "baseline"),
    # training.py lives at the repository root.
    top_level=("training",),
    # Imported at module level by training.py and baseline/*.py, used only when logging
    # to Weights & Biases (`--log`), which the adapter never enables.
    stubs={"wandb": ("init", "log", "finish", "Table")},
)

#: The official configuration closest to our task: one label per answer (tokenwise:
#: false), short QA answers, attention graph + activations ("att_act"). The NQ and CNN
#: configurations are token-level (span annotations), which our labels are not.
CONFIG = "configs/charm/movies_att_act.yaml"

#: data_prep_2/data_collection.py: --att_threshold 0.05, prompt_graph off (README).
GRAPH_KWARGS = {"threshold": 0.05, "prompt_graph": False}

#: README collects activations of layers 24,28,32 and the configs train on "24", of
#: 32-layer models (Llama-2-7B, Mistral-7B): 0.75 of the depth (METHOD_CARD.md, D2).
ACT_DEPTH_FRACTION = 0.75


def act_layer(n_layers: int) -> int:
    """Hidden-state index (0 = embeddings) of the activation layer, at 0.75 of the depth."""
    return max(1, min(n_layers, round(ACT_DEPTH_FRACTION * n_layers)))


def load() -> dict:
    mods = load_upstream(SPEC, ["dataset.graphs", "dataset.acts", "dataset.transforms",
                                "model.mp", "training"])
    return {name.rsplit(".", 1)[-1]: module for name, module in mods.items()}
