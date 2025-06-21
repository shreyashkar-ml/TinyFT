__version__ = "0.1.0"

from .adapters import LoRAAdapter, QLoRAAdapter, AdapterBase
from .manager import AdapterManager
from .trainer import TinyFTTrainer
from .datasets import SFTDataset, CPTDataset, DatasetBuilder
from .utils import setup_logging, freeze_parameters, get_model_info

from .engine import vLLMEngine, SGLangEngine

__all__ = [
    # Adapters
    "LoRAAdapter",
    "QLoRAAdapter", 
    "AdapterBase",
    
    # Manager
    "AdapterManager",
    
    # Training
    "TinyFTTrainer",
    
    # Datasets
    "SFTDataset",
    "CPTDataset", 
    "DatasetBuilder",
    
    # Utilities
    "setup_logging",
    "freeze_parameters",
    "get_model_info",
    
    # Inference
    "vLLMEngine",
    "SGLangEngine",
]

__author__ = "Shreyashkar"
__email__ = "ishreyashkar06@gmail.com"
__license__ = "MIT"
__description__ = "Lightweight fine-tuning library for large language models" 