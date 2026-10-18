"""Pretrained text-to-text adapter boundary for future Phase 10 experiments."""

from __future__ import annotations


class PretrainedSeq2SeqUnavailable(RuntimeError):
    """Raised when optional pretrained translation dependencies are unavailable."""


class PretrainedSeq2SeqTranslator:
    """Thin adapter that fails clearly unless optional dependencies are installed."""

    def __init__(self, model_name: str, *, task_prefix: str = "") -> None:
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # type: ignore
        except ImportError as error:  # pragma: no cover - dependency optional
            raise PretrainedSeq2SeqUnavailable(
                "Install the 'transformers' optional dependency to use pretrained_seq2seq."
            ) from error
        self.model_name = model_name
        self.task_prefix = task_prefix
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def generate(self, gloss_text: str, *, max_new_tokens: int = 64, num_beams: int = 1) -> str:
        text = f"{self.task_prefix}{gloss_text}" if self.task_prefix else gloss_text
        inputs = self.tokenizer(text, return_tensors="pt")
        output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, num_beams=num_beams)
        return str(self.tokenizer.decode(output_ids[0], skip_special_tokens=True))
