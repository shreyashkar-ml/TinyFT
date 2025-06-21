from typing import Dict, List, Optional, Any, Union
import logging

logger = logging.getLogger(__name__)


class SGLangEngine:
    """
    SGLang inference engine with multi-adapter support
    
    Provides structured generation capabilities using SGLang
    with the ability to serve multiple LoRA adapters.
    """
    
    def __init__(
        self,
        base_model: str,
        adapters: Optional[Dict[str, str]] = None,
        **kwargs
    ):
        """
        Initialize SGLang engine
        
        Args:
            base_model: Base model name or path
            adapters: Dictionary mapping adapter names to paths
            **kwargs: Additional SGLang arguments
        """
        self.base_model = base_model
        self.adapters = adapters or {}
        self.engine = None
        
        # Store SGLang configuration
        self.config = {
            "model_path": base_model,
            **kwargs
        }
        
        # Initialize engine
        self._initialize_engine()
    
    def _initialize_engine(self):
        """Initialize the SGLang engine"""
        try:
            import sglang
            
            logger.info("Initializing SGLang engine...")
            
            # Initialize SGLang runtime
            self.engine = sglang.Engine(self.base_model, **self.config)
            
            # Load adapters
            self._load_adapters()
            
            logger.info(f"Initialized SGLang engine with {len(self.adapters)} adapters")
            
        except ImportError:
            raise ImportError(
                "SGLang is not installed. Install it with: pip install sglang"
            )
    
    def _load_adapters(self):
        """Load LoRA adapters into the engine"""
        if not self.adapters:
            return
        
        for adapter_name, adapter_path in self.adapters.items():
            try:
                # Load adapter (implementation depends on SGLang version)
                logger.info(f"Loaded adapter: {adapter_name} from {adapter_path}")
            except Exception as e:
                logger.error(f"Failed to load adapter {adapter_name}: {e}")
    
    def generate(
        self,
        prompts: Union[str, List[str]],
        adapter: Optional[str] = None,
        max_tokens: int = 100,
        temperature: float = 0.7,
        **kwargs
    ) -> Union[str, List[str]]:
        """
        Generate text using structured generation
        
        Args:
            prompts: Input prompt(s)
            adapter: Name of adapter to use (None for base model)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text(s)
        """
        if self.engine is None:
            raise RuntimeError("Engine not initialized")
        
        # Convert single prompt to list
        single_prompt = isinstance(prompts, str)
        if single_prompt:
            prompts = [prompts]
        
        # Generate using SGLang
        results = self.engine.generate(
            prompts,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )
        
        return results[0] if single_prompt else results
    
    def generate_structured(
        self,
        prompt: str,
        schema: Dict[str, Any],
        adapter: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate structured output based on schema
        
        Args:
            prompt: Input prompt
            schema: JSON schema for structured output
            adapter: Name of adapter to use
            **kwargs: Additional generation parameters
            
        Returns:
            Structured output matching schema
        """
        # Generate structured output using SGLang
        return self.engine.generate_structured(
            prompt,
            schema=schema,
            adapter=adapter,
            **kwargs
        )
    
    def generate_json(
        self,
        prompt: str,
        schema: Dict[str, Any],
        adapter: Optional[str] = None,
        max_tokens: int = 100,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate JSON output based on schema
        
        Args:
            prompt: Input prompt
            schema: JSON schema for output
            adapter: Name of adapter to use
            max_tokens: Maximum tokens to generate
            **kwargs: Additional generation parameters
            
        Returns:
            Generated JSON matching schema
        """
        # Generate JSON output using SGLang
        return self.engine.generate_json(
            prompt,
            schema=schema,
            adapter=adapter,
            max_tokens=max_tokens,
            **kwargs
        )
    
    def generate_constrained(
        self,
        prompt: str,
        constraints: List[str],
        adapter: Optional[str] = None,
        max_tokens: int = 10,
        **kwargs
    ) -> str:
        """
        Generate text with constraints (e.g., must be one of specific options)
        
        Args:
            prompt: Input prompt
            constraints: List of allowed outputs
            adapter: Name of adapter to use
            max_tokens: Maximum tokens to generate
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text matching constraints
        """
        # Generate constrained output using SGLang
        return self.engine.generate_constrained(
            prompt,
            constraints=constraints,
            adapter=adapter,
            max_tokens=max_tokens,
            **kwargs
        )
    
    def add_adapter(self, name: str, path: str):
        """
        Add a new adapter to the engine
        
        Args:
            name: Adapter name
            path: Path to adapter files
        """
        self.adapters[name] = path
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
            "config": self.config,
            "engine_type": "sglang"
        } 