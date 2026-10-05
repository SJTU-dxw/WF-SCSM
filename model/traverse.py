from __future__ import annotations

import torch
from torch import nn


class TraVerseMLM(nn.Module):
    """Masked reconstruction wrapper; Transformers/PEFT are optional dependencies."""

    def __init__(self, config: dict):
        super().__init__()
        try:
            from transformers import AutoModelForCausalLM
            from peft import LoraConfig, get_peft_model
        except ImportError as exc:
            raise RuntimeError("TraVerse requires: pip install transformers peft") from exc
        model = AutoModelForCausalLM.from_pretrained(
            config["pretrained_model"],
            trust_remote_code=bool(config.get("trust_remote_code", False)),
            torch_dtype=getattr(torch, config.get("torch_dtype", "float32")),
        )
        if bool(config.get("bidirectional_attention", True)):
            # Recent Transformers versions let decoder-only models use a full
            # bidirectional padding mask when this configuration flag is false.
            # The shifted causal-LM loss is retained for masked reconstruction.
            model.config.is_causal = False
            model.config.use_cache = False
        if bool(config.get("gradient_checkpointing", True)):
            model.gradient_checkpointing_enable()
            model.config.use_cache = False
        lora = config["lora"]
        self.model = get_peft_model(model, LoraConfig(
            r=int(lora["rank"]), lora_alpha=int(lora["alpha"]),
            lora_dropout=float(lora["dropout"]), bias="none",
            task_type="CAUSAL_LM", target_modules=lora.get("target_modules"),
        ))

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.model(**batch).loss
