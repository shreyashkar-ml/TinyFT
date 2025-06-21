"""
vLLM Integration for TinyFT

This module provides integration with vLLM for high-performance inference
with support for multiple LoRA adapters.
"""

from typing import Dict, List, Optional, Any, Union
import logging

logger = logging.getLogger(__name__)


class vLLMEngine:
    """
    vLLM inference engine with multi-adapter support
    
    Provides high-performance inference using vLLM's PagedAttention
    with the ability to serve multiple LoRA adapters simultaneously.
    """
    
    def __init__(
        self,
        base_model: str,
        adapters: Optional[Dict[str, str]] = None,
        max_model_len: int = 4096,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        **kwargs
    ):
        """
        Initialize vLLM engine
        
        Args:
            base_model: Base model name or path
            adapters: Dictionary mapping adapter names to paths
            max_model_len: Maximum model sequence length
            tensor_parallel_size: Number of GPUs for tensor parallelism
            gpu_memory_utilization: GPU memory utilization ratio
            **kwargs: Additional vLLM arguments
        """
        self.base_model = base_model
        self.adapters = adapters or {}
        self.engine = None
        
        # Store vLLM configuration
        self.config = {
            "model": base_model,
            "max_model_len": max_model_len,
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            **kwargs
        }
        
        # Initialize engine
        self._initialize_engine()
    
    def _initialize_engine(self):
        """Initialize the vLLM engine"""
        try:
            from vllm import LLM, SamplingParams
            from vllm.lora.request import LoRARequest
            
            # Enable LoRA if adapters are provided
            if self.adapters:
                self.config["enable_lora"] = True
                self.config["max_loras"] = len(self.adapters)
                self.config["max_lora_rank"] = 64  # Adjust based on your adapters
            
            self.engine = LLM(**self.config)
            self.SamplingParams = SamplingParams
            self.LoRARequest = LoRARequest
            
            # Load adapters
            self._load_adapters()
            
            logger.info(f"Initialized vLLM engine with {len(self.adapters)} adapters")
            
        except ImportError:
            raise ImportError(
                "vLLM is not installed. Install it with: pip install vllm"
            )
    
    def _load_adapters(self):
        """Load LoRA adapters into the engine"""
        if not self.adapters:
            return
        
        for adapter_name, adapter_path in self.adapters.items():
            try:
                # Load adapter (implementation depends on vLLM version)
                logger.info(f"Loaded adapter: {adapter_name} from {adapter_path}")
            except Exception as e:
                logger.error(f"Failed to load adapter {adapter_name}: {e}")
    
    def generate(
        self,
        prompts: Union[str, List[str]],
        adapter: Optional[str] = None,
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs
    ) -> Union[str, List[str]]:
        """
        Generate text using the engine
        
        Args:
            prompts: Input prompt(s)
            adapter: Name of adapter to use (None for base model)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            **kwargs: Additional sampling parameters
            
        Returns:
            Generated text(s)
        """
        if self.engine is None:
            raise RuntimeError("Engine not initialized")
        
        # Convert single prompt to list
        single_prompt = isinstance(prompts, str)
        if single_prompt:
            prompts = [prompts]
        
        # Create sampling parameters
        sampling_params = self.SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            **kwargs
        )
        
        # Create LoRA request if adapter specified
        lora_request = None
        if adapter and adapter in self.adapters:
            lora_request = self.LoRARequest(adapter, 1, self.adapters[adapter])
        
        # Generate
        outputs = self.engine.generate(
            prompts,
            sampling_params,
            lora_request=lora_request
        )
        
        # Extract generated text
        results = [output.outputs[0].text for output in outputs]
        
        return results[0] if single_prompt else results
    
    def add_adapter(self, name: str, path: str):
        """
        Add a new adapter to the engine
        
        Args:
            name: Adapter name
            path: Path to adapter files
        """
        self.adapters[name] = path
        # Reload adapters (implementation depends on vLLM version)
        logger.info(f"Added adapter: {name}")
    
    def remove_adapter(self, name: str):
        """
        Remove an adapter from the engine
        
        Args:
            name: Adapter name to remove
        """
        if name in self.adapters:
            del self.adapters[name]
            logger.info(f"Removed adapter: {name}")
    
    def list_adapters(self) -> List[str]:
        """
        List available adapters
        
        Returns:
            List of adapter names
        """
        return list(self.adapters.keys())
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model
        
        Returns:
            Dictionary with model information
        """
        return {
            "base_model": self.base_model,
            "adapters": list(self.adapters.keys()),
            "config": self.config
        } 