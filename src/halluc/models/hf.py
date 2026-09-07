"""HuggingFace causal-LM generator.

Two forward regimes per item, deliberately:

1. `generate()` under SDPA — fast, and attention weights are not needed yet.
2. one re-forward of prompt+answer under eager attention — yields the full [T, T]
   attention matrix the spectral features are defined over.

Re-forwarding rather than harvesting attentions during generation is exact under greedy
decoding (the re-forward is teacher-forced on the tokens the model actually produced)
and avoids stitching together the ragged per-step attention slices that `generate`
emits. The extra pass costs one forward against 256 decode steps.
"""

from __future__ import annotations

import time
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def _load_causal_lm(model_id: str, **kwargs):
    """Load a generator, picking the auto class by architecture.

    Gemma 3 ships as Gemma3ForConditionalGeneration (multimodal) even though we use it
    text-only, so it does not load under AutoModelForCausalLM.
    """
    architectures = getattr(AutoConfig.from_pretrained(model_id), "architectures", None) or []
    if any("ConditionalGeneration" in a or "ImageText" in a for a in architectures):
        from transformers import AutoModelForImageTextToText

        return AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
    return AutoModelForCausalLM.from_pretrained(model_id, **kwargs)


#: Generators validated with this pipeline. The registry accepts any HF causal LM, so
#: this is documentation and a `--model` shorthand, not a restriction. Each entry's
#: quirks are handled automatically: Gemma 3 loads through the image-text-to-text auto
#: class, and its sliding-window layers are absorbed by LapEigvals' measured out-degree.
KNOWN_GENERATORS = {
    "qwen3-14b": "Qwen/Qwen3-14B",
    "qwen3-4b": "Qwen/Qwen3-4B-Instruct-2507",
    "llama3.2-3b": "meta-llama/Llama-3.2-3B-Instruct",
    "gemma3-4b": "google/gemma-3-4b-it",
    "gemma3-12b": "google/gemma-3-12b-it",
}


def resolve_model_id(name: str) -> str:
    """Accept either a preset shorthand or a full HF model id."""
    return KNOWN_GENERATORS.get(name.lower(), name)


def model_dims(model_id: str) -> dict:
    """Layer/head/hidden sizes, reading `text_config` for multimodal checkpoints."""
    cfg = AutoConfig.from_pretrained(model_id)
    text = getattr(cfg, "text_config", None) or cfg
    return {
        "n_layers": text.num_hidden_layers,
        "n_heads": text.num_attention_heads,
        "hidden_size": text.hidden_size,
        # Gemma 3 interleaves sliding-window layers; see LapEigvals' divisor handling.
        "sliding_window": getattr(text, "sliding_window", None),
        "architecture": (architectures[0] if (architectures := cfg.architectures) else None),
    }

from ..datasets.base import QAItem
from ..features.base import ForwardTrace
from ..prompts import BASE_STOP_STRINGS, base_prompt_text, generator_messages
from .base import GENERATORS, Generation, Generator


@GENERATORS.register("hf_causal")
class HFGenerator(Generator):
    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-14B",
        device: str = "cuda:0",
        dtype: str = "bfloat16",
        max_new_tokens: int = 256,
        enable_thinking: bool = False,
        max_seq_len: int = 4096,
        load_in_4bit: bool = False,
        reserve_gib: float = 6.0,
    ) -> None:
        self.name = model_id
        self.model_id = model_id
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.enable_thinking = enable_thinking
        # Attention memory is 40*40*T^2*2 bytes across all layers: ~8 GB at T=1571
        # (CoQA's longest), on top of the 28 GB model. The cap is a guard against a
        # pathological item OOMing a multi-hour run.
        self.max_seq_len = max_seq_len

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        kwargs: dict = {"dtype": getattr(torch, dtype), "attn_implementation": "sdpa"}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        self._require_memory(device, load_in_4bit, reserve_gib, model_id)
        if load_in_4bit:
            kwargs["device_map"] = {"": device}
        self.model = _load_causal_lm(model_id, **kwargs)
        if not load_in_4bit:
            self.model.to(device)
        self.model.eval()
        self.input_device = device
        self.dims = model_dims(model_id)

    @staticmethod
    def _weight_gib(model_id: str) -> float:
        """Size of the checkpoint on disk, in GiB.

        Measured rather than assumed: generators here span 3B to 14B, so a fixed figure
        would either wave through an OOM or refuse a GPU that is actually big enough.
        """
        try:
            from huggingface_hub import snapshot_download

            path = Path(snapshot_download(model_id, local_files_only=True))
            total = sum(f.stat().st_size for f in path.glob("*.safetensors"))
            if total:
                return total / 1024**3
        except Exception:  # noqa: BLE001  (cache miss, network off, layout change)
            pass
        return 28.0  # conservative fallback: the largest generator used here

    @classmethod
    def _require_memory(cls, device: str, load_in_4bit: bool, reserve_gib: float,
                        model_id: str = "") -> None:
        """Fail fast with a readable message instead of a mid-load CUDA OOM traceback.

        This machine is shared, so a GPU that was free when the run was configured may
        not be by the time it starts.
        """
        if not device.startswith("cuda"):
            return
        index = int(device.split(":")[1]) if ":" in device else 0
        free_gib = torch.cuda.mem_get_info(index)[0] / 1024**3
        weights = cls._weight_gib(model_id) if model_id else 28.0
        # 4-bit stores roughly a quarter of the bf16 weights.
        needed = weights / 3.5 if load_in_4bit else weights
        if free_gib < needed + reserve_gib:
            raise RuntimeError(
                f"{device} has {free_gib:.1f} GiB free but this run needs about "
                f"{needed:.0f} GiB of weights plus {reserve_gib:.0f} GiB of headroom. "
                f"Free the GPU, point --device at another one, or set load_in_4bit."
            )

    @property
    def completion_mode(self) -> bool:
        """True for base checkpoints: no chat template, so few-shot completion instead."""
        return not getattr(self.tokenizer, "chat_template", None)

    def _prompt_text(self, item: QAItem) -> str:
        if self.completion_mode:
            return base_prompt_text(item)
        return self.tokenizer.apply_chat_template(
            generator_messages(item),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )

    @torch.inference_mode()
    def generate(self, item: QAItem) -> tuple[Generation, ForwardTrace]:
        started = time.perf_counter()
        prompt = self._prompt_text(item)
        prompt_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.input_device)
        prompt_len = prompt_ids.shape[1]

        gen_kwargs: dict = {}
        if self.completion_mode:
            # A base model does not emit EOS after an answer; it invents the next Q/A
            # pair. Without a stop condition the run-on text would move the final token
            # position, which is exactly what the hidden-state probes read.
            gen_kwargs = {"stop_strings": BASE_STOP_STRINGS, "tokenizer": self.tokenizer}

        out = self.model.generate(
            prompt_ids,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            **gen_kwargs,
        )
        full_ids = out[0]
        answer_ids = full_ids[prompt_len:]
        # Trailing EOS/pad are not part of the answer and would skew the last-token
        # features, which are read from the final position of the sequence.
        keep = len(answer_ids)
        while keep > 0 and answer_ids[keep - 1].item() in self._stop_ids():
            keep -= 1
        hit_cap = len(answer_ids) >= self.max_new_tokens and keep == len(answer_ids)
        finish_reason = "length" if hit_cap else "stop"
        answer_ids = answer_ids[:keep]

        answer = self.tokenizer.decode(answer_ids, skip_special_tokens=True)
        if self.completion_mode:
            keep, answer = self._trim_completion(answer_ids, answer, keep)
            answer_ids = answer_ids[:keep]
        seq = full_ids[: prompt_len + keep].unsqueeze(0)
        if seq.shape[1] > self.max_seq_len:
            raise ValueError(f"sequence {seq.shape[1]} exceeds max_seq_len {self.max_seq_len}")

        trace = self._trace(seq, prompt_len)
        generation = Generation(
            item_id=item.item_id,
            answer=answer,
            prompt_tokens=prompt_len,
            answer_tokens=int(keep),
            finish_reason=finish_reason,
            seconds=time.perf_counter() - started,
            meta={"seq_len": int(seq.shape[1])},
        )
        return generation, trace

    @torch.inference_mode()
    def retrace(
        self, item: QAItem, answer: str, expected_tokens: int | None = None
    ) -> ForwardTrace:
        """Forward trace for an answer stage 1 already produced, without regenerating.

        CHARM needs a full attention trace per item but no new text: stage 1 recorded
        what the model said, and greedy decoding makes that reproducible. Re-encoding
        the stored answer and running the single eager forward costs one pass instead of
        up to 256 decode steps, which is the difference between hours and a day over
        four datasets.

        The one risk is a tokenizer round trip that does not land back on the same ids
        (measured at ~0.1% of items here, from decoded text that re-encodes one token
        shorter). `expected_tokens` makes that visible: a mismatch raises, and the caller
        falls back to a real `generate()` for that item rather than silently tracing a
        sequence the model never produced.
        """
        prompt = self._prompt_text(item)
        prompt_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.input_device)
        prompt_len = prompt_ids.shape[1]
        answer_ids = self.tokenizer(
            answer, return_tensors="pt", add_special_tokens=False
        ).input_ids.to(self.input_device)

        if expected_tokens is not None and answer_ids.shape[1] != expected_tokens:
            raise ValueError(
                f"re-encoded answer is {answer_ids.shape[1]} tokens, stage 1 recorded "
                f"{expected_tokens}"
            )

        seq = torch.cat([prompt_ids, answer_ids], dim=1)
        if seq.shape[1] > self.max_seq_len:
            raise ValueError(f"sequence {seq.shape[1]} exceeds max_seq_len {self.max_seq_len}")
        return self._trace(seq, prompt_len)

    def _trim_completion(self, answer_ids, answer: str, keep: int) -> tuple[int, str]:
        """Cut a base model's answer at the first stop marker, in tokens as well as text.

        `stop_strings` halts generation but leaves the marker in the output, and anything
        after the answer proper shifts the final token — the position every hidden-state
        probe reads. Trimming the string alone would fix the label and corrupt the
        features, so the token count is trimmed to match and verified: a tokenizer round
        trip that does not land on a clean prefix leaves the answer untouched rather than
        silently truncating to the wrong position.
        """
        cuts = [answer.find(m) for m in BASE_STOP_STRINGS]
        cuts = [c for c in cuts if c >= 0]
        target = (answer[: min(cuts)] if cuts else answer).strip()
        if not target or target == answer.strip():
            return keep, answer.strip()
        lo, hi = 0, keep
        while lo < hi:                      # smallest prefix whose decode covers target
            mid = (lo + hi) // 2
            if self.tokenizer.decode(answer_ids[:mid], skip_special_tokens=True).strip() \
                    .startswith(target):
                hi = mid
            else:
                lo = mid + 1
        got = self.tokenizer.decode(answer_ids[:lo], skip_special_tokens=True).strip()
        return (lo, got) if got == target else (keep, answer.strip())

    def _stop_ids(self) -> set[int]:
        ids = {self.tokenizer.eos_token_id, self.tokenizer.pad_token_id}
        return {i for i in ids if i is not None}

    @torch.inference_mode()
    def _trace(self, seq: torch.Tensor, prompt_len: int) -> ForwardTrace:
        """Single eager forward over prompt+answer, yielding [H, T, T] attentions."""
        self.model.set_attn_implementation("eager")
        try:
            out = self.model(
                seq, output_attentions=True, output_hidden_states=True, use_cache=False
            )
            return ForwardTrace(
                attentions=tuple(a[0] for a in out.attentions),
                hidden_states=tuple(h[0] for h in out.hidden_states),
                prompt_len=prompt_len,
            )
        finally:
            self.model.set_attn_implementation("sdpa")

    def unload(self) -> None:
        del self.model
        torch.cuda.empty_cache()
