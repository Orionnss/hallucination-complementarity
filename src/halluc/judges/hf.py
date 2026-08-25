"""HuggingFace judge, quantised to 4-bit NF4.

Quantisation is safe here in a way it is not for the generator: a judge only has to emit
one of three label tokens, and nothing downstream reads its activations.

Judges are loaded one at a time and unloaded before the next, so a three-model pool fits
on a single GPU alongside nothing else.
"""

from __future__ import annotations

import time

import torch
from transformers import AutoTokenizer, BitsAndBytesConfig

from ..prompts import judge_messages
from .base import JUDGES, Judge, Label, parse_label


def _load_model(model_id: str, device: str, load_in_4bit: bool):
    """Pick the auto class by architecture.

    gemma-3-12b-it is a Gemma3ForConditionalGeneration (multimodal) checkpoint, so it
    does not load under AutoModelForCausalLM even though we use it text-only.
    """
    from transformers import AutoConfig, AutoModelForCausalLM

    kwargs: dict = {"dtype": torch.bfloat16}
    if load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        kwargs["device_map"] = {"": device}

    architectures = getattr(AutoConfig.from_pretrained(model_id), "architectures", None) or []
    if any("ConditionalGeneration" in a or "ImageText" in a for a in architectures):
        from transformers import AutoModelForImageTextToText

        model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    if not load_in_4bit:
        model.to(device)
    return model.eval()


@JUDGES.register("hf_judge")
class HFJudge(Judge):
    def __init__(
        self,
        model_id: str = "google/gemma-3-12b-it",
        device: str = "cuda:0",
        load_in_4bit: bool = True,
        max_new_tokens: int = 8,
    ) -> None:
        self.name = model_id
        self.model_id = model_id
        self.device = device
        self.max_new_tokens = max_new_tokens
        tokenizer_kwargs: dict = {}
        if "mistral" in model_id.lower():
            # Without this the shipped tokenizer uses an incorrect regex and tokenises
            # the prompt wrongly; transformers warns about it on load.
            tokenizer_kwargs["fix_mistral_regex"] = True
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, **tokenizer_kwargs)
        self.model = _load_model(model_id, device, load_in_4bit)

    @torch.inference_mode()
    def judge(self, question: str, gold_answers: list[str], answer: str) -> tuple[Label, str]:
        started = time.perf_counter()
        messages = judge_messages(question, gold_answers, answer)
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        out = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )
        raw = self.tokenizer.decode(
            out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        self.last_seconds = time.perf_counter() - started
        return parse_label(raw), raw.strip()

    def unload(self) -> None:
        del self.model
        torch.cuda.empty_cache()
