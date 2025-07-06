"""Utilities for reward-based training using the GRPO algorithm."""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from typing import Any, Dict, List

from .trainer import TinyFTTrainer


class GRPODataset(Dataset):
    """Simple dataset for GRPO training.

    Each sample should contain a ``prompt`` text, a ``response`` text and a
    numeric ``reward``. The ``prompt`` and ``response`` are concatenated and
    tokenized similar to ``SFTDataset``.
    """

    def __init__(
        self,
        data: List[Dict[str, Any]],
        tokenizer,
        max_length: int = 512,
        prompt_field: str = "prompt",
        response_field: str = "response",
        reward_field: str = "reward",
    ) -> None:
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt_field = prompt_field
        self.response_field = response_field
        self.reward_field = reward_field

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.data[idx]
        prompt = sample[self.prompt_field]
        response = sample[self.response_field]
        reward = float(sample[self.reward_field])

        text = prompt + response
        tok = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        labels = tok["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": tok["input_ids"].squeeze(0),
            "attention_mask": tok["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0),
            "reward": torch.tensor(reward, dtype=torch.float),
        }


class GRPOTrainer(TinyFTTrainer):
    """Minimal GRPO trainer based on :class:`TinyFTTrainer`."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, training_type="grpo", **kwargs)

    def _compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:  # type: ignore[override]
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)
        rewards = batch["reward"].to(self.device)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        logits = outputs.logits
        vocab = logits.size(-1)
        ce = F.cross_entropy(
            logits.view(-1, vocab),
            labels.view(-1),
            reduction="none",
            ignore_index=-100,
        )
        ce = ce.view(labels.size())
        logp = -(ce * (labels != -100)).sum(dim=1)
        loss = -(rewards * logp).mean()
        return loss
