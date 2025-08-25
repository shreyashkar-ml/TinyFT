import unittest
from typing import Any, Dict, List

import torch
import torch.nn as nn

from pathlib import Path
import sys

# Ensure package import
sys.path.insert(0, str(Path(__file__).parent.parent))

from tinyft import TinyGRPOTrainer  # type: ignore


class DummyTokenizer:
    def __init__(self) -> None:
        self.pad_token_id = 0
        self.eos_token_id = 1
        # Fixed small vocab of lowercase letters + space
        alphabet = " " + "abcdefghijklmnopqrstuvwxyz"
        self.char_to_id = {ch: i + 2 for i, ch in enumerate(alphabet)}
        self.id_to_char = {i + 2: ch for i, ch in enumerate(alphabet)}
        # Reserve 0 for pad, 1 for eos

    def __call__(
        self,
        texts: List[str],
        add_special_tokens: bool = False,
        padding: bool = False,
        truncation: bool = False,
        return_tensors: Any = None,
    ) -> Dict[str, List[List[int]]]:
        input_ids: List[List[int]] = []
        for t in texts:
            ids = [self.char_to_id.get(ch, self.char_to_id[" "]) for ch in t]
            if add_special_tokens:
                ids = ids + [self.eos_token_id]
            input_ids.append(ids)
        return {"input_ids": input_ids}

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        out = []
        for i in ids:
            if i == self.pad_token_id:
                continue
            if i == self.eos_token_id and skip_special_tokens:
                continue
            out.append(self.id_to_char.get(i, " "))
        return "".join(out)


class DummyCausalLM(nn.Module):
    def __init__(self, vocab_size: int = 64, hidden: int = 16) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden)
        self.proj = nn.Linear(hidden, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # input_ids: [B, T]
        emb = self.embed(input_ids)  # [B, T, H]
        logits = self.proj(emb)  # [B, T, V]
        return logits


def simple_reward(response: str, prompt: str, context: Dict[str, Any]) -> Dict[str, float]:
    # Reward if letter 'a' appears in response
    r = 1.0 if ("a" in response) else 0.0
    return {"reward": r, "reward_info": {"has_a": r}}


class TestTinyGRPOTrainer(unittest.TestCase):
    def test_grpo_trainer_runs(self) -> None:
        tokenizer = DummyTokenizer()

        # Vocab size from tokenizer (ids go up to len(alphabet)+1)
        vocab_size = max(list(tokenizer.id_to_char.keys())) + 1
        model = DummyCausalLM(vocab_size=vocab_size, hidden=8)

        prompts = ["hello", "world"]

        trainer = TinyGRPOTrainer(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            reward_fn=simple_reward,
            max_gen_len=5,
            num_questions_per_batch=2,
            num_answers_per_question=2,
            micro_batch_size=2,
            total_steps=2,
            fp16=False,
            bf16=False,
            logging_steps=1,
            output_dir="./outputs_test",
        )

        metrics = trainer.train()

        self.assertIn("loss", metrics)
        self.assertIn("entropy", metrics)
        self.assertIn("grad_norm", metrics)
        self.assertTrue(isinstance(metrics["loss"], float))


if __name__ == "__main__":
    unittest.main()

