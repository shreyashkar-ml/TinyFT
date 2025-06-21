import os
import logging
import time
from typing import Dict, Any, Optional, Union
from pathlib import Path
import torch
import torch.nn as nn


def setup_logging(
    backend: str = "tensorboard",
    project_name: str = "tinyft_experiment",
    run_name: Optional[str] = None,
    log_dir: str = "./logs",
    log_level: str = "INFO"
) -> Dict[str, Any]:
    """
    Setup logging backends for training
    
    Args:
        backend: Logging backend ("tensorboard", "wandb", "both")
        project_name: Name of the project/experiment
        run_name: Name of the specific run (auto-generated if None)
        log_dir: Directory for logs
        log_level: Python logging level
        
    Returns:
        Dictionary with logger instances
    """
    # Setup Python logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger("tinyft")
    
    # Create log directory
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Generate run name if not provided
    if run_name is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_name = f"{project_name}_{timestamp}"
    
    loggers = {"python": logger}
    
    # Setup TensorBoard
    if backend in ["tensorboard", "both"]:
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_log_dir = log_path / "tensorboard" / run_name
            tb_logger = SummaryWriter(tb_log_dir)
            loggers["tensorboard"] = tb_logger
            logger.info(f"TensorBoard logging to: {tb_log_dir}")
        except ImportError:
            logger.warning("TensorBoard not available, skipping")
    
    # Setup Weights & Biases
    if backend in ["wandb", "both"]:
        try:
            import wandb
            wandb.init(
                project=project_name,
                name=run_name,
                dir=log_dir
            )
            loggers["wandb"] = wandb
            logger.info(f"Weights & Biases logging initialized")
        except ImportError:
            logger.warning("Weights & Biases not available, skipping")
    
    return loggers


def freeze_parameters(model: nn.Module, patterns: Optional[list] = None) -> int:
    """
    Freeze model parameters based on patterns
    
    Args:
        model: Model to freeze parameters in
        patterns: List of parameter name patterns to freeze (None = freeze all)
        
    Returns:
        Number of parameters frozen
    """
    frozen_count = 0
    
    for name, param in model.named_parameters():
        should_freeze = False
        
        if patterns is None:
            # Freeze all parameters
            should_freeze = True
        else:
            # Check if parameter matches any pattern
            for pattern in patterns:
                if pattern in name:
                    should_freeze = True
                    break
        
        if should_freeze:
            param.requires_grad = False
            frozen_count += 1
    
    return frozen_count


def unfreeze_parameters(model: nn.Module, patterns: Optional[list] = None) -> int:
    """
    Unfreeze model parameters based on patterns
    
    Args:
        model: Model to unfreeze parameters in
        patterns: List of parameter name patterns to unfreeze (None = unfreeze all)
        
    Returns:
        Number of parameters unfrozen
    """
    unfrozen_count = 0
    
    for name, param in model.named_parameters():
        should_unfreeze = False
        
        if patterns is None:
            # Unfreeze all parameters
            should_unfreeze = True
        else:
            # Check if parameter matches any pattern
            for pattern in patterns:
                if pattern in name:
                    should_unfreeze = True
                    break
        
        if should_unfreeze:
            param.requires_grad = True
            unfrozen_count += 1
    
    return unfrozen_count


def get_model_info(model: nn.Module) -> Dict[str, Any]:
    """
    Get comprehensive information about a model
    
    Args:
        model: Model to analyze
        
    Returns:
        Dictionary with model information
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Get model size in MB
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    model_size_mb = (param_size + buffer_size) / (1024 ** 2)
    
    # Get device information
    devices = {str(p.device) for p in model.parameters() if p.device.type != 'meta'}
    
    info = {
        "model_class": model.__class__.__name__,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "trainable_percentage": (trainable_params / total_params) * 100 if total_params > 0 else 0,
        "model_size_mb": model_size_mb,
        "devices": list(devices),
        "dtype": str(next(model.parameters()).dtype) if list(model.parameters()) else "unknown"
    }
    
    return info


def estimate_memory_usage(
    model: nn.Module,
    batch_size: int = 1,
    sequence_length: int = 2048,
    bytes_per_param: int = 2
) -> Dict[str, float]:
    """
    Estimate memory usage for training
    
    Args:
        model: Model to estimate for
        batch_size: Training batch size
        sequence_length: Input sequence length
        bytes_per_param: Bytes per parameter (2 for FP16, 4 for FP32)
        
    Returns:
        Dictionary with memory estimates in GB
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Model weights
    model_memory = (total_params * bytes_per_param) / (1024 ** 3)
    
    # Gradients (only for trainable parameters)
    grad_memory = (trainable_params * bytes_per_param) / (1024 ** 3)
    
    # Optimizer states (Adam: 2x gradients)
    optimizer_memory = grad_memory * 2
    
    # Activations (rough estimate based on batch size and sequence length)
    activation_memory = (batch_size * sequence_length * 1024 * bytes_per_param) / (1024 ** 3)
    
    total_memory = model_memory + grad_memory + optimizer_memory + activation_memory
    
    return {
        "model_memory_gb": model_memory,
        "gradient_memory_gb": grad_memory,
        "optimizer_memory_gb": optimizer_memory,
        "activation_memory_gb": activation_memory,
        "total_estimated_gb": total_memory
    }


def format_time(seconds: float) -> str:
    """
    Format time duration into human-readable string
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        seconds = seconds % 60
        return f"{int(minutes)}m {seconds:.0f}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{int(hours)}h {int(minutes)}m"


def format_number(num: Union[int, float]) -> str:
    """
    Format large numbers with appropriate suffixes
    
    Args:
        num: Number to format
        
    Returns:
        Formatted number string
    """
    if num >= 1_000_000_000:
        return f"{num / 1_000_000_000:.1f}B"
    elif num >= 1_000_000:
        return f"{num / 1_000_000:.1f}M"
    elif num >= 1_000:
        return f"{num / 1_000:.1f}K"
    else:
        return str(num)


def create_progress_format(
    examples_trained: int,
    total_examples: int,
    epoch: int,
    total_epochs: int,
    it_per_sec: float,
    loss: Optional[float] = None,
    lr: Optional[float] = None
) -> str:
    """
    Create formatted progress string for tqdm
    
    Args:
        examples_trained: Number of examples processed
        total_examples: Total examples in dataset
        epoch: Current epoch
        total_epochs: Total epochs
        it_per_sec: Iterations per second
        loss: Current loss value
        lr: Current learning rate
        
    Returns:
        Formatted progress string
    """
    progress_str = f"---- {examples_trained}/{total_examples} Epoch: {epoch}/{total_epochs}, {it_per_sec:.1f} it/s"
    
    if loss is not None:
        progress_str += f" | Loss: {loss:.4f}"
    
    if lr is not None:
        progress_str += f" | LR: {lr:.2e}"
    
    return progress_str


def save_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    step: int,
    loss: float,
    save_path: Union[str, Path],
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Save training checkpoint
    
    Args:
        model: Model to save
        optimizer: Optimizer to save (can be None)
        epoch: Current epoch
        step: Current step
        loss: Current loss
        save_path: Path to save checkpoint
        metadata: Additional metadata to save
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "step": step,
        "loss": loss,
        "metadata": metadata or {}
    }
    
    # Only save optimizer state if optimizer is provided
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    
    torch.save(checkpoint, save_path)


def load_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_path: Union[str, Path]
) -> Dict[str, Any]:
    """
    Load training checkpoint
    
    Args:
        model: Model to load state into
        optimizer: Optimizer to load state into
        checkpoint_path: Path to checkpoint file
        
    Returns:
        Dictionary with checkpoint metadata
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    return {
        "epoch": checkpoint.get("epoch", 0),
        "step": checkpoint.get("step", 0),
        "loss": checkpoint.get("loss", 0.0),
        "metadata": checkpoint.get("metadata", {})
    }


def get_device(prefer_cuda: bool = True) -> torch.device:
    """
    Get the best available device
    
    Args:
        prefer_cuda: Whether to prefer CUDA over other devices
        
    Returns:
        PyTorch device
    """
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility
    
    Args:
        seed: Random seed value
    """
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class MemoryTracker:
    """Track memory usage during training"""
    
    def __init__(self, device: torch.device):
        self.device = device
        self.peak_memory = 0
        
    def update(self) -> float:
        """Update and return current memory usage in MB"""
        if self.device.type == "cuda":
            current = torch.cuda.memory_allocated(self.device) / (1024 ** 2)
            self.peak_memory = max(self.peak_memory, current)
            return current
        return 0.0
    
    def reset_peak(self) -> None:
        """Reset peak memory tracking"""
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        self.peak_memory = 0 