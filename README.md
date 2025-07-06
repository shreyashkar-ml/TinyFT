# TinyFT: Lightweight Fine-Tuning Library

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

TinyFT is a lightweight, modular fine-tuning library designed for parameter-efficient fine-tuning of large language models. Built as a minimal implementation of `peft` and `unsloth` production-grade library, it provides unified APIs for LoRA, QLoRA, and full fine-tuning with minimal dependencies.

## Key Features

- **Parameter-Efficient Fine-Tuning**: LoRA and QLoRA implementations with automatic target module detection
- **Unified Training Interface**: Single API for supervised fine-tuning (SFT) and continued pre-training (CPT)
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

### GRPO Training

```python
from tinyft import GRPODataset, GRPOTrainer

# Create dataset with rewards
data = [
    {"prompt": "Say hi", "response": "Hi there!", "reward": 1.0},
    {"prompt": "Say bye", "response": "Goodbye", "reward": 0.5},
]
dataset = GRPODataset(data, tokenizer)

# Train using GRPO
trainer = GRPOTrainer(model=model, dataset=dataset, batch_size=2)
trainer.train()
```

## TODO:

1. ~~Add GRPO Training functionality.~~ ✅ Implemented via `GRPOTrainer`.
2. Add custom CUDA / triton quantization solution where PyTorch native quantization API fails instead of using `FakeQuantize`.

## License

This project is licensed under the MIT License.

## Acknowledgments

- [Hugging Face](https://huggingface.co/) for the transformers library
- [Microsoft LoRA](https://github.com/microsoft/LoRA) for the original LoRA implementation
- [QLoRA](https://github.com/artidoro/qlora) for quantized LoRA techniques
- [vLLM](https://github.com/vllm-project/vllm) for high-performance inference