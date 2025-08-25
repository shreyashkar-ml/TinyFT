import torch
import torch.nn as nn
from typing import List, Optional, Dict, Any, Union, Tuple
import logging
from pathlib import Path
import json

from .adapters import LoRAAdapter, QLoRAAdapter, AdapterBase, create_adapter


logger = logging.getLogger(__name__)


class AdapterManager:
    """
    High-level adapter lifecycle management
    
    Handles automatic target module detection, adapter application,
    and management across different model architectures.
    """
    
    # Default target modules for different model types
    DEFAULT_TARGET_MODULES = {
        "llama": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "mistral": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], 
        "qwen": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "gpt2": ["c_attn", "c_proj", "c_fc"],
        "gpt_neox": ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"], 
        "bert": ["query", "key", "value", "dense"],
        "roberta": ["query", "key", "value", "dense"],
        "t5": ["q", "k", "v", "o", "wi_0", "wi_1", "wo"],
        "gemma": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    }
    
    def __init__(self):
        self.applied_adapters: Dict[str, Dict[str, Any]] = {}
        self.model_info: Optional[Dict[str, Any]] = None
    
    def detect_model_type(self, model: nn.Module) -> str:
        """
        Automatically detect model architecture type
        
        Args:
            model: The model to analyze
            
        Returns:
            Detected model type string
        """
        model_class = model.__class__.__name__.lower()
        
        # Check for common model types
        for model_type in self.DEFAULT_TARGET_MODULES.keys():
            if model_type in model_class:
                logger.info(f"Detected model type: {model_type}")
                return model_type
        
        # If not found, try to infer from model structure
        module_names = [name for name, _ in model.named_modules()]
        
        # Look for characteristic module patterns
        if any("q_proj" in name for name in module_names):
            logger.info("Inferred model type: llama (based on q_proj modules)")
            return "llama"
        elif any("c_attn" in name for name in module_names):
            logger.info("Inferred model type: gpt2 (based on c_attn modules)")
            return "gpt2"
        elif any("query" in name for name in module_names):
            logger.info("Inferred model type: bert (based on query modules)")
            return "bert"
        
        logger.warning("Could not detect model type, using generic target modules")
        return "unknown"
    
    def get_default_target_modules(self, model_type: str) -> List[str]:
        """
        Get default target modules for a model type
        
        Args:
            model_type: The model type identifier
            
        Returns:
            List of default target module names
        """
        # Try exact match first
        if model_type in self.DEFAULT_TARGET_MODULES:
            return self.DEFAULT_TARGET_MODULES[model_type].copy()
        
        # Try partial matches
        for key, modules in self.DEFAULT_TARGET_MODULES.items():
            if key in model_type.lower():
                return modules.copy()
        
        # Fallback to common attention modules
        logger.warning(f"No default modules for {model_type}, using fallback")
        return ["q_proj", "k_proj", "v_proj", "o_proj"]
    
    def find_target_modules(
        self, 
        model: nn.Module, 
        target_modules: Union[str, List[str]]
    ) -> Dict[str, nn.Module]:
        """
        Find target modules in the model
        
        Args:
            model: The model to search
            target_modules: Module names to find or "auto" for auto-detection
            
        Returns:
            Dictionary mapping module paths to modules
        """
        if isinstance(target_modules, str):
            if target_modules.lower() == "auto":
                model_type = self.detect_model_type(model)
                target_modules = self.get_default_target_modules(model_type)
            else:
                target_modules = [target_modules]
        
        found_modules = {}
        
        for name, module in model.named_modules():
            # Check for both Linear and Conv1D layers (Conv1D is used in GPT-2)
            if isinstance(module, nn.Linear):
                module_name = name.split('.')[-1]
                if module_name in target_modules:
                    found_modules[name] = module
            elif hasattr(module, 'weight') and hasattr(module, 'bias') and len(module.weight.shape) == 2:
                # This handles Conv1D layers which have 2D weight tensors like Linear layers
                module_name = name.split('.')[-1]
                if module_name in target_modules:
                    found_modules[name] = module
        
        if not found_modules:
            # Get all available linear modules for debugging
            available_modules = [
                name.split('.')[-1] 
                for name, module in model.named_modules() 
                if isinstance(module, nn.Linear)
            ]
            raise ValueError(
                f"No target modules found. Target: {target_modules}, "
                f"Available: {set(available_modules)}"
            )
        
        logger.info(f"Found {len(found_modules)} target modules: {list(found_modules.keys())}")
        return found_modules
    
    def apply_adapters(
        self,
        model: nn.Module,
        method: str = "lora",
        target_modules: Union[str, List[str]] = "auto",
        model_name: Optional[str] = None,
        **adapter_kwargs
    ) -> nn.Module:
        """
        Apply adapters to a model
        
        Args:
            model: The base model to adapt
            method: Adapter method ("lora", "qlora", "full")
            target_modules: Target modules or "auto" for auto-detection
            model_name: Optional model name for loading from HuggingFace
            **adapter_kwargs: Additional arguments for adapter creation
            
        Returns:
            Model with adapters applied
        """
        if method not in ["lora", "qlora", "full"]:
            raise ValueError(f"Unsupported method: {method}. Use 'lora', 'qlora', or 'full'")
        
        # Handle model loading if model_name is provided
        if model_name is not None:
            model = self._load_model_from_name(model_name)
        
        if method == "full":
            # Full fine-tuning - unfreeze all parameters
            for param in model.parameters():
                param.requires_grad = True
            logger.info("Applied full fine-tuning (all parameters trainable)")
            return model
        
        # Apply PEFT adapters
        target_modules_dict = self.find_target_modules(model, target_modules)
        
        # Apply adapters to each target module
        for module_path, linear_module in target_modules_dict.items():
            adapter = create_adapter(
                linear_module,
                adapter_type=method,
                **adapter_kwargs
            )
            
            # Replace the module in the model
            self._replace_module(model, module_path, adapter)
            
            # Track applied adapters
            self.applied_adapters[module_path] = {
                "method": method,
                "original_module": linear_module,
                "adapter": adapter,
                "config": adapter_kwargs.copy()
            }
        
        logger.info(f"Applied {method.upper()} adapters to {len(target_modules_dict)} modules")
        return model
    
    def _load_model_from_name(self, model_name: str) -> nn.Module:
        """Load model from HuggingFace model name"""
        try:
            from transformers import AutoModel, AutoModelForCausalLM
            
            # Try different model types
            try:
                model = AutoModelForCausalLM.from_pretrained(model_name)
            except:
                model = AutoModel.from_pretrained(model_name)
            
            logger.info(f"Loaded model: {model_name}")
            return model
            
        except ImportError:
            raise ImportError("transformers library required for loading models by name")
    
    def _replace_module(self, model: nn.Module, module_path: str, new_module: nn.Module):
        """Replace a module in the model at the given path"""
        path_parts = module_path.split('.')
        parent = model
        
        for part in path_parts[:-1]:
            parent = getattr(parent, part)
        
        setattr(parent, path_parts[-1], new_module)
    
    def merge_adapters(self, model: nn.Module) -> nn.Module:
        """
        Merge all applied adapters into base weights
        
        Args:
            model: Model with adapters to merge
            
        Returns:
            Model with merged adapters
        """
        merged_count = 0
        
        for module_path, adapter_info in self.applied_adapters.items():
            adapter = adapter_info["adapter"]
            if hasattr(adapter, 'merge'):
                adapter.merge()
                merged_count += 1
        
        logger.info(f"Merged {merged_count} adapters into base weights")
        return model
    
    def unmerge_adapters(self, model: nn.Module) -> None:
        """
        Unmerge all applied adapters from base weights
        
        Args:
            model: Model with merged adapters to unmerge
        """
        unmerged_count = 0
        
        for module_path, adapter_info in self.applied_adapters.items():
            adapter = adapter_info["adapter"]
            if hasattr(adapter, 'unmerge'):
                adapter.unmerge()
                unmerged_count += 1
        
        logger.info(f"Unmerged {unmerged_count} adapters from base weights")
    
    def save_adapters(self, model: nn.Module, save_path: Union[str, Path]) -> None:
        """
        Save adapter weights and configuration
        
        Args:
            model: Model with adapters
            save_path: Path to save adapters
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        adapter_weights = {}
        adapter_configs = {}
        
        for module_path, adapter_info in self.applied_adapters.items():
            adapter = adapter_info["adapter"]
            
            # Save adapter weights
            if isinstance(adapter, (LoRAAdapter, QLoRAAdapter)):
                adapter_weights[module_path] = {
                    'A': adapter.A.data,
                    'B': adapter.B.data,
                    'scaling': adapter.scaling
                }
            
            # Save adapter config
            adapter_configs[module_path] = {
                "method": adapter_info["method"],
                "config": adapter_info["config"]
            }
        
        # Save weights
        torch.save(adapter_weights, save_path / "adapter_weights.pth")
        
        # Save configuration
        with open(save_path / "adapter_config.json", 'w') as f:
            json.dump(adapter_configs, f, indent=2)
        
        # Save model info if available
        if self.model_info:
            with open(save_path / "model_info.json", 'w') as f:
                json.dump(self.model_info, f, indent=2)
        
        logger.info(f"Saved {len(adapter_weights)} adapters to {save_path}")
    
    def load_adapters(self, model: nn.Module, load_path: Union[str, Path]) -> None:
        """
        Load adapter weights from saved checkpoint
        
        Args:
            model: Model to load adapters into
            load_path: Path to load adapters from
        """
        load_path = Path(load_path)
        
        # Load weights
        weights_path = load_path / "adapter_weights.pth"
        if not weights_path.exists():
            raise FileNotFoundError(f"Adapter weights not found: {weights_path}")
        
        adapter_weights = torch.load(weights_path, map_location='cpu')
        
        # Load configuration
        config_path = load_path / "adapter_config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                adapter_configs = json.load(f)
        else:
            logger.warning("Adapter config not found, using default")
            adapter_configs = {}
        
        # Apply weights to model
        loaded_count = 0
        for module_path, weights in adapter_weights.items():
            # Find the adapter module in the current model
            try:
                module = model
                for part in module_path.split('.'):
                    module = getattr(module, part)
                
                if hasattr(module, 'A') and hasattr(module, 'B'):
                    module.A.data = weights['A']
                    module.B.data = weights['B']
                    if 'scaling' in weights:
                        module.scaling = weights['scaling']
                    loaded_count += 1
                    
            except AttributeError:
                logger.warning(f"Could not find adapter module: {module_path}")
        
        logger.info(f"Loaded {loaded_count} adapters from {load_path}")
    
    def get_trainable_parameters(self, model: nn.Module) -> Dict[str, int]:
        """
        Get statistics about trainable parameters
        
        Args:
            model: Model to analyze
            
        Returns:
            Dictionary with parameter statistics
        """
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        
        stats = {
            "trainable": trainable,
            "total": total,
            "percentage": (trainable / total) * 100 if total > 0 else 0,
            "adapters": len(self.applied_adapters)
        }
        
        # Add adapter-specific stats
        if self.applied_adapters:
            adapter_params = 0
            for adapter_info in self.applied_adapters.values():
                adapter = adapter_info["adapter"]
                adapter_params += sum(p.numel() for p in adapter.parameters() if p.requires_grad)
            
            stats["adapter_params"] = adapter_params
        
        return stats
    
    def get_adapter_summary(self) -> Dict[str, Any]:
        """
        Get summary of all applied adapters
        
        Returns:
            Dictionary with adapter summary information
        """
        if not self.applied_adapters:
            return {"message": "No adapters applied"}
        
        summary = {
            "total_adapters": len(self.applied_adapters),
            "adapters": {}
        }
        
        for module_path, adapter_info in self.applied_adapters.items():
            adapter = adapter_info["adapter"]
            adapter_summary = {
                "method": adapter_info["method"],
                "is_merged": adapter.is_merged if hasattr(adapter, 'is_merged') else False,
                "config": adapter_info["config"]
            }
            
            if isinstance(adapter, (LoRAAdapter, QLoRAAdapter)):
                adapter_summary.update({
                    "rank": adapter.r,
                    "alpha": adapter.alpha,
                    "scaling": adapter.scaling
                })
            
            summary["adapters"][module_path] = adapter_summary
        
        return summary 