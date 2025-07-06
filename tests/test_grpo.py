import unittest
import torch
import torch.nn as nn
from tinyft.grpo_trainer import GRPODataset, GRPOTrainer

class SimpleLM(nn.Module):
    def __init__(self, vocab_size=10, hidden=8):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden)
        self.fc = nn.Linear(hidden, vocab_size)

    def forward(self, input_ids, attention_mask=None, labels=None):
        x = self.embed(input_ids)
        logits = self.fc(x)
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100
            )
        return type("Out", (), {"logits": logits, "loss": loss})()


class TestGRPO(unittest.TestCase):
    def setUp(self):
        self.tokenizer = type("Tok", (), {
            "pad_token": None,
            "eos_token": "",
            "pad_token_id": 0,
            "__call__": lambda self, text, **kw: {
                "input_ids": torch.tensor([[1,2,3,0]]),
                "attention_mask": torch.tensor([[1,1,1,0]])
            }
        })()
        self.model = SimpleLM()
        self.data = [
            {"prompt": "A", "response": "B", "reward": 1.0},
            {"prompt": "C", "response": "D", "reward": 0.5},
        ]

    def test_dataset(self):
        ds = GRPODataset(self.data, self.tokenizer)
        sample = ds[0]
        self.assertIn("reward", sample)
        self.assertEqual(sample["input_ids"].shape[0], 4)

    def test_trainer_step(self):
        ds = GRPODataset(self.data, self.tokenizer)
        trainer = GRPOTrainer(model=self.model, dataset=ds, batch_size=2, num_epochs=1, logging_backend="tensorboard", logging_steps=1)
        batch = next(iter(torch.utils.data.DataLoader(ds, batch_size=2)))
        loss = trainer._compute_loss(batch)
        self.assertTrue(torch.is_tensor(loss))


if __name__ == "__main__":
    unittest.main()
