"""Run configuration. Every value here is stamped into each stage's JSON."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class GeneratorConfig:
    kind: str = "hf_causal"
    model_id: str = "Qwen/Qwen3-14B"
    #: Single GPU, pinned. No sharding.
    device: str = "cuda:1"
    #: bf16, not 4-bit: the probes read these activations, and quantisation error would
    #: land directly in the features under study. Judges are quantised instead.
    dtype: str = "bfloat16"
    load_in_4bit: bool = False
    max_new_tokens: int = 256
    enable_thinking: bool = False
    max_seq_len: int = 4096
    #: GiB held back per GPU for attention tensors and activations.
    reserve_gib: float = 6.0


@dataclass
class JudgeConfig:
    kind: str = "hf_judge"
    model_id: str = "google/gemma-3-12b-it"
    device: str = "cuda:0"
    #: Judges only need to emit a label, so NF4 costs nothing that matters here and
    #: lets a 3-model pool run sequentially on one GPU.
    load_in_4bit: bool = True
    max_new_tokens: int = 8
    #: Judging is ~280 prompt tokens and 8 generated tokens per item, so
    #: unbatched it leaves most of the GPU idle.
    batch_size: int = 32


@dataclass
class CharmConfig:
    """CHARM (arXiv 2509.24770). Not a stage-1 feature block — see stage6_charm.

    CHARM's input is a whole attention graph per sample, which is too large and too
    ragged to persist alongside the other methods' `.npz` blocks, so it builds its
    graphs in its own extraction loop and trains from a RAM cache.
    """

    #: Attention sparsification threshold of Equation 1. The paper sweeps
    #: {0.5, 0.1, 0.05, 0.01, 0.001} and keeps 0.05 as the best accuracy/footprint
    #: trade-off.
    tau: float = 0.05
    #: Residual-stream layers to attach as node features, as fractions of model depth.
    #: Empty list gives the attention-only variant, CHARM (att).
    act_fractions: list[float] = field(default_factory=lambda: [0.7])
    #: "default" = the pinned four-point grid; "full" = Table 9's 288-point search space.
    grid: str = "default"
    #: Bounds the dense [E, hidden] message tensor when a batch collects several long
    #: CoQA graphs.
    max_edges_per_batch: int = 400_000
    #: Abort rather than swap if the RAM graph cache would exceed this.
    max_cache_gib: float = 300.0


@dataclass
class Config:
    run_id: str = "main"
    output_root: str = "runs"
    datasets: list[str] = field(default_factory=lambda: ["triviaqa", "nq_open", "squad_v2", "coqa"])
    pool_size: int = 4000
    pool_seed: int = 0
    #: Samples drawn from the pool per experiment seed.
    n_per_seed: int = 2000
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    n_folds: int = 5
    #: "pooled" trains one probe on every dataset and evaluates it per dataset;
    #: "per_dataset" trains and evaluates each dataset independently.
    training_scope: str = "pooled"
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    judges: list[JudgeConfig] = field(
        default_factory=lambda: [
            JudgeConfig(model_id="google/gemma-3-12b-it"),
            JudgeConfig(model_id="Qwen/Qwen2.5-14B-Instruct"),
            # Nemo over-flags relative to the other two (45.8% HALLUCINATED vs gemma
            # 23.2% / qwen 29.8%), and as the least-agreeing judge it decides contested
            # items toward HALLUCINATED ~90% of the time. It is kept because the
            # alternatives tried were worse: Llama-3.1-8B (3.7% HALLUCINATED) and
            # Llama-3.2-3B (0.2%) barely flag anything, and scored lower kappa against
            # the gemma+qwen consensus on every dataset. Their verdicts remain on disk.
            # Report the over-flagging bias as a limitation; see DESIGN.md.
            JudgeConfig(model_id="mistralai/Mistral-Nemo-Instruct-2407"),
        ]
    )
    features: list[str] = field(
        default_factory=lambda: ["lapeigvals", "attn_baseline", "saplma", "svd_baseline", "icr"]
    )
    shard_size: int = 250
    charm: CharmConfig = field(default_factory=CharmConfig)

    @property
    def run_dir(self) -> Path:
        return Path(self.output_root) / self.run_id

    def stage_dir(self, stage: str, *parts: str) -> Path:
        return self.run_dir.joinpath(stage, *parts)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def load(cls, path: str | Path | None) -> "Config":
        if path is None:
            return cls()
        with open(path) as fh:
            raw = yaml.safe_load(fh) or {}
        gen = GeneratorConfig(**raw.pop("generator", {}))
        charm = CharmConfig(**raw.pop("charm", {}))
        judges = [JudgeConfig(**j) for j in raw.pop("judges", [])] or None
        cfg = cls(generator=gen, charm=charm, **raw)
        if judges:
            cfg.judges = judges
        return cfg
