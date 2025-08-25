# TinyFT: Lightweight Fine-Tuning Library

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

TinyFT is a lightweight, modular fine-tuning library designed for parameter-efficient fine-tuning of large language models. Built as a minimal implementation of `peft` and `unsloth` production-grade library, it provides unified APIs for LoRA, QLoRA, and full fine-tuning with minimal dependencies.

## Key Features

- **Parameter-Efficient Fine-Tuning**: LoRA and QLoRA implementations with automatic target module detection
- **Unified Training Interface**: Single API for supervised fine-tuning (SFT), continued pre-training (CPT), and GRPO (RL)
- **Multiple Logging Backends**: TensorBoard and Weights & Biases support
- **Memory Optimization**: Gradient checkpointing, mixed precision, and quantization support
- **High-Performance Inference**: vLLM and SGLang integration for multi-adapter serving
- **Clean Architecture**: Modular design with minimal dependencies

## Installation

```bash
# Install directly from GitHub
pip install git+https://github.com/shreyashkar-ml/TinyFT.git

# Or clone and install locally
git clone https://github.com/shreyashkar-ml/TinyFT.git
cd tinyft
pip install -e .

# This will automatically install all required dependencies
```

## Architecture

```
tinyft/
├── __init__.py           # Main package exports
├── adapters.py           # LoRA & QLoRA implementations
├── manager.py            # AdapterManager: load/unload/merge adapters
├── trainer.py            # Unified trainer for SFT & continued pre-training
├── datasets.py           # Dataset wrappers and utilities
├── utils.py              # Logging, parameter freezing, misc helpers
├── engine/               # High-performance inference backends
│   ├── __init__.py       # Engine exports
│   ├── vllm_engine.py    # vLLM integration for fast inference
│   └── sglang_engine.py  # SGLang integration
└── scripts/              # Command-line scripts
    ├── train_sft.py      # Supervised fine-tuning entry point
    ├── train_pretrain.py # Continued pre-training entry point
    └── train_grpo.py     # GRPO training entry point
    └── merge_adapters.py # Merge adapters into base model
configs/
├── base_lora.yaml        # LoRA configuration example
├── base_qlora.yaml       # QLoRA configuration example
└── trainer_sft.yaml      # Training configuration example
```

## Quick Start

### Basic LoRA Fine-Tuning

```python
from tinyft import AdapterManager, TinyFTTrainer, SFTDataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")

# Apply LoRA adapters
manager = AdapterManager()
model = manager.apply_adapters(
    model=model,
    method="lora",
    target_modules=["q_proj", "v_proj"],
    r=8,
    alpha=16
)

# Prepare dataset
data = [
    {"instruction": "What is AI?", "output": "AI is artificial intelligence."},
    {"instruction": "What is 2+2?", "output": "2+2 equals 4."}
]
dataset = SFTDataset.from_json(data, tokenizer, max_length=128)

# Train
trainer = TinyFTTrainer(
    model=model,
    training_type="sft",
    dataset=dataset,
    learning_rate=1e-4,
    batch_size=1,
    max_steps=10
)
trainer.train()
```

### GRPO CLI Usage

Run GRPO via the CLI using YAML configs.

- Basic run (full precision):
  - `tinyft-train-grpo --config configs/grpo_example.yaml`

- 4-bit run (bitsandbytes):
  - Ensure `bitsandbytes` is installed in your environment.
  - Export allocator tuning for better stability:
    - `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb=64`
  - Then run:
    - `tinyft-train-grpo --config configs/grpo_4bit_example.yaml`

Supported YAML fields (subset):
- `model.model_name`, `model.tokenizer_name` (optional)
- `model.quantization.*` (optional): `load_in_4bit`, `bnb_4bit_quant_type`, `bnb_4bit_use_double_quant`, `bnb_4bit_compute_dtype`
- `model.attn_implementation` (e.g., `flash_attention_2`), `model.device_map` (e.g., `auto`)
- `model.torch_dtype` (`bfloat16`, `float16`, etc.), `model.use_cache`, `model.gradient_checkpointing`
- `training.*` as shown in `configs/grpo_example.yaml` and `configs/grpo_4bit_example.yaml`

### QLoRA for Memory Efficiency

```python
# Apply QLoRA for quantized training
model = manager.apply_adapters(
    model=model,
    method="qlora",
    target_modules=["q_proj", "v_proj"],
    r=4,
    alpha=8,
    quant_bits=4
)
```

### Command Line Usage

```bash
# Basic SFT training
python -m tinyft.scripts.train_sft --config configs/base_lora.yaml

# Continued pre-training
python -m tinyft.scripts.train_pretrain --config configs/base_lora_cpt.yaml

# Merge adapters
python -m tinyft.scripts.merge_adapters --adapter_path ./outputs --output_path ./merged_model

# GRPO training
python -m tinyft.scripts.train_grpo --config configs/grpo_example.yaml
## Or as an installed CLI
tinyft-train-grpo --config configs/grpo_example.yaml
```

## GRPO Training

TinyFT includes a lightweight Group Relative Policy Optimization (GRPO) trainer for RL-style preference/reward optimization without heavy dependencies.

Example usage:

```python
from tinyft import TinyGRPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load a small causal LM (or your fine-tuned one)
model = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2")
tokenizer = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")

# Define prompts and a reward function
prompts = [
    {"prompt": "Write a friendly greeting."},
    {"prompt": "Explain GRPO in one sentence."},
]

def reward_fn(response: str, prompt: str, context: dict):
    # Example: reward longer and polite responses
    score = 0.0
    score += 0.1 * min(len(response.split()), 50)
    if any(w in response.lower() for w in ["please", "thank", "kind"]):
        score += 1.0
    return {"reward": score}

trainer = TinyGRPOTrainer(
    model=model,
    tokenizer=tokenizer,
    prompts=prompts,
    reward_fn=reward_fn,
    max_gen_len=64,
    num_questions_per_batch=2,
    num_answers_per_question=2,
    total_steps=100,
    fp16=False,  # set True if running on GPU
    bf16=True,
    output_dir="./outputs/grpo_run",
)

trainer.train()
```

Notes:
- The trainer computes token-wise advantages per group of answers and performs a single policy update per sampling iteration.
- Provide your own reward function for your task; return either a float or `{ "reward": float, "reward_info": {...} }`.
- For a minimal runnable example without external downloads, see `tests/run_grpo_example.py`.
- Use the CLI with `tinyft-train-grpo --config <yaml>` for YAML-driven runs; see `configs/grpo_example.yaml`.

**TODO (future improvement)**: Refactor TinyGRPOTrainer sampling/generation into a pluggable interface so it works seamlessly with any model’s generation API (beyond plain forward logits), while keeping current behavior stable.

### GRPO Memory Tuning (35 GB GPUs)

To reliably run GRPO with 4B-class models on ~35–40 GB GPUs, use the settings below. Recent changes make these memory optimizations automatic where possible.

- Env var: set once before training for better allocator behavior.
  - `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb=64`
- Optimizer: if `bitsandbytes` is installed, the trainer uses 8-bit (paged) AdamW automatically; otherwise falls back to torch AdamW with `foreach=False` to reduce fused allocations.
- KV cache: disabled during training in the trainer to save memory; no user action needed.
- AMP: logits forward uses autocast (bf16/fp16) in the policy update; set `bf16=True` or `fp16=True` in training args.
- Checkpointing: enabled if the model supports it; no user action needed.
- Recommended GRPO knobs (tune for your model/dataset):
  - `max_gen_len`: 256–512
  - `num_questions_per_batch` (Q): 4–8
  - `num_answers_per_question` (M): 1–2
  - `micro_batch_size`: 2–4
  - `max_grad_norm`: 0.5–1.0

YAML example for the GRPO CLI (configs/grpo_example.yaml):

```yaml
training:
  max_gen_len: 256
  num_questions_per_batch: 4
  num_answers_per_question: 1
  micro_batch_size: 2
  total_steps: 100
  learning_rate: 1e-5
  max_grad_norm: 0.5
  fp16: false     # set true on non-Ampere GPUs
  bf16: true      # prefer true on Ampere+ GPUs
```

Optional: load 4-bit base + apply adapters (Python API)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tinyft import AdapterManager, TinyGRPOTrainer

model_name = "Qwen/Qwen2.5-3B"  # example 3–4B class model

bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)
tok = AutoTokenizer.from_pretrained(model_name)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_cfg)

# Apply (Q)LoRA adapters to attention/MLP modules
manager = AdapterManager()
model = manager.apply_adapters(
    model=model,
    method="lora",           # or "qlora" (requires bitsandbytes)
    target_modules="auto",   # auto-detect target modules per architecture
    r=16,
    alpha=32,
    dropout=0.0,
    # quant_bits=4            # if method="qlora"
)

trainer = TinyGRPOTrainer(
    model=model,
    tokenizer=tok,
    prompts=[{"prompt": "Explain GRPO in short."}],
    reward_fn=lambda r, p, c: {"reward": float(len(r.split()) > 5)},
    max_gen_len=256,
    num_questions_per_batch=4,
    num_answers_per_question=1,
    micro_batch_size=2,
    total_steps=10,
    bf16=True,
)
trainer.train()
```

### Multi-Adapter Inference

```python
from tinyft.engine import vLLMEngine

# Setup inference engine
engine = vLLMEngine(
    base_model="facebook/opt-125m",
    max_model_len=2048
)

# Generate text
response = engine.generate(
    prompts=["Hello, how are you?"],
    max_tokens=50,
    temperature=0.7
)
print(response[0])
```

## Configuration

TinyFT uses YAML configuration files for reproducible experiments. Example configurations are provided in the `configs/` directory.

### LoRA Configuration Example

```yaml
# Basic LoRA Settings
r: 8                           # Rank of low-rank decomposition
alpha: 16                      # Scaling factor
dropout: 0.1                   # Dropout probability
target_modules: ["q_proj", "v_proj"]  # Target modules for adaptation

# Training Settings
learning_rate: 1e-4
batch_size: 1
max_steps: 100
```

## Advanced Features

### Memory Optimization

```python
# Enable gradient checkpointing and mixed precision
trainer = TinyFTTrainer(
    model=model,
    gradient_checkpointing=True,
    fp16=True,
    gradient_accumulation_steps=4
)
```

### Custom Datasets

```python
# Load from HuggingFace datasets
dataset = SFTDataset.from_hf(
    "yahma/alpaca-cleaned",
    tokenizer=tokenizer,
    max_length=512
)

# Load from text file
dataset = SFTDataset.from_text_file(
    "data.jsonl",
    tokenizer=tokenizer
)
```

### Logging Integration

```python
# Use TensorBoard or Weights & Biases
trainer = TinyFTTrainer(
    model=model,
    logging_backend="tensorboard",  # or "wandb" or "both"
    logging_steps=10
)
```

## TODO:

1. Add custom CUDA / triton quantization solution where PyTorch native quantization API fails instead of using `FakeQuantize`.

## License

This project is licensed under the MIT License.

## Acknowledgments

- [Hugging Face](https://huggingface.co/) for the transformers library
- [Microsoft LoRA](https://github.com/microsoft/LoRA) for the original LoRA implementation
- [QLoRA](https://github.com/artidoro/qlora) for quantized LoRA techniques
- [vLLM](https://github.com/vllm-project/vllm) for high-performance inference
