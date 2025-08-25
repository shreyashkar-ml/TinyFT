import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any


class AdapterBase(nn.Module, ABC):
    """
    Abstract base class for all adapters
    
    Provides common interface and functionality for different adapter types.
    """
    
    def __init__(self, base_layer: nn.Module):
        super().__init__()
        self.base_layer = base_layer
        self._merged = False
        
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the adapter"""
        pass
    
    @abstractmethod
    def merge(self) -> None:
        """Merge adapter weights into base layer"""
        pass
    
    @abstractmethod
    def unmerge(self) -> None:
        """Unmerge adapter weights from base layer"""
        pass
    
    @property
    def is_merged(self) -> bool:
        return self._merged


class LoRAAdapter(AdapterBase):
    """
    Low-Rank Adaptation (LoRA) implementation
    
    Decomposes weight updates into two low-rank matrices A and B,
    significantly reducing the number of trainable parameters.
    
    Args:
        base_layer: The base linear layer to adapt
        r: Rank of the low-rank decomposition
        alpha: Scaling factor for LoRA weights
        dropout: Dropout probability for LoRA layers
        init_method: Weight initialization method ('kaiming', 'normal', 'zero')
    """
    
    def __init__(
        self,
        base_layer: nn.Module,  # Support both nn.Linear and Conv1D layers
        r: int = 8,
        alpha: int = 8,
        dropout: float = 0.0,
        init_method: str = "kaiming"
    ):
        super().__init__(base_layer)
        
        # Check layer type first
        is_conv1d = (hasattr(base_layer, 'weight') and 
                    hasattr(base_layer, 'bias') and 
                    len(base_layer.weight.shape) == 2 and
                    hasattr(base_layer, 'nf'))  # Conv1D has 'nf' attribute
        
        # Check if it's a supported layer type (Linear or Conv1D)
        if not isinstance(base_layer, nn.Linear) and not is_conv1d:
            raise ValueError("LoRAAdapter only supports nn.Linear or Conv1D layers")
        
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        
        # Get dimensions from base layer
        if is_conv1d:
            # For Conv1D, weight shape is (d_in, d_out)
            d_in, d_out = base_layer.weight.shape
        else:
            # For Linear, weight shape is (d_out, d_in)
            d_out, d_in = base_layer.weight.shape
        
        # LoRA matrices
        self.A = nn.Parameter(torch.zeros(r, d_in))
        self.B = nn.Parameter(torch.zeros(d_out, r))
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        
        # Initialize weights
        self._init_weights(init_method)
        
        # Freeze base layer parameters
        self._freeze_base_layer()
    
    def _is_conv1d_layer(self, layer: nn.Module) -> bool:
        """Check if the layer is a Conv1D layer (like in GPT-2)"""
        return (hasattr(layer, 'weight') and 
                hasattr(layer, 'bias') and 
                len(layer.weight.shape) == 2 and
                hasattr(layer, 'nf'))  # Conv1D has 'nf' attribute
    
    def _init_weights(self, method: str) -> None:
        """Initialize LoRA weights"""
        if method == "kaiming":
            nn.init.kaiming_uniform_(self.A, a=0, mode='fan_in')
            nn.init.zeros_(self.B)
        elif method == "normal":
            nn.init.normal_(self.A, std=0.02)
            nn.init.zeros_(self.B)
        elif method == "zero":
            nn.init.zeros_(self.A)
            nn.init.zeros_(self.B)
        else:
            raise ValueError(f"Unknown initialization method: {method}")
    
    def _freeze_base_layer(self) -> None:
        """Freeze base layer parameters"""
        for param in self.base_layer.parameters():
            param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through base layer + LoRA adaptation"""
        base_output = self.base_layer(x)
        
        # If the adapter is merged, return the base output
        if self._merged:
            return base_output
        
        # LoRA computation - both Conv1D and Linear use same pattern
        # Conv1D in transformers: output = input @ weight + bias
        # Linear: output = input @ weight.T + bias
        # But both can use the same LoRA computation pattern
        
        # Ensure LoRA matrices are on the same device as input
        A_device = self.A.to(x.device)
        B_device = self.B.to(x.device)
        
        # Standard LoRA computation: x @ (B @ A).T
        x_reshaped = x.view(-1, x.size(-1))  # Flatten to 2D
        lora_delta = (B_device @ A_device).T  # (d_in, d_out)
        lora_output = self.scaling * (self.dropout(x_reshaped) @ lora_delta)
        lora_output = lora_output.view(*x.size()[:-1], -1)  # Restore shape
        
        return base_output + lora_output
    
    def merge(self) -> None:
        """Merge LoRA weights into base layer weights"""
        if self._merged:
            return
        
        # Compute delta weight: scaling * B @ A
        A_device = self.A.to(self.base_layer.weight.device)
        B_device = self.B.to(self.base_layer.weight.device)
        delta_weight = self.scaling * (B_device @ A_device)
        
        # For Conv1D layers, we need to transpose the delta weight
        # because Conv1D weight shape is (d_in, d_out) vs Linear (d_out, d_in)
        if self._is_conv1d_layer(self.base_layer):
            delta_weight = delta_weight.T
        
        # Add to base weights
        self.base_layer.weight.data += delta_weight
        self._merged = True
    
    def unmerge(self) -> None:
        """Remove LoRA weights from base layer weights"""
        if not self._merged:
            return
        
        # Compute delta weight: scaling * B @ A
        A_device = self.A.to(self.base_layer.weight.device)
        B_device = self.B.to(self.base_layer.weight.device)
        delta_weight = self.scaling * (B_device @ A_device)
        
        # For Conv1D layers, we need to transpose the delta weight
        # because Conv1D weight shape is (d_in, d_out) vs Linear (d_out, d_in)
        if self._is_conv1d_layer(self.base_layer):
            delta_weight = delta_weight.T
        
        # Subtract from base weights
        self.base_layer.weight.data -= delta_weight
        self._merged = False
    
    def get_delta_weight(self) -> torch.Tensor:
        """Get the delta weight matrix (B @ A)"""
        A_device = self.A.to(self.B.device)
        return self.scaling * (self.B @ A_device)
    
    def extra_repr(self) -> str:
        return f"r={self.r}, alpha={self.alpha}, scaling={self.scaling:.3f}"


class QLoRAAdapter(LoRAAdapter):
    """
    Quantized LoRA (QLoRA) implementation
    
    Combines LoRA with quantization of the base layer to further reduce
    memory usage during training.
    
    Args:
        base_layer: The base linear layer to adapt
        r: Rank of the low-rank decomposition  
        alpha: Scaling factor for LoRA weights
        dropout: Dropout probability for LoRA layers
        quant_bits: Number of bits for quantization (4 or 8)
        init_method: Weight initialization method
    """
    
    def __init__(
        self,
        base_layer: nn.Module,  # Changed from nn.Linear to nn.Module to support Conv1D
        r: int = 8,
        alpha: int = 8,
        dropout: float = 0.0,
        quant_bits: int = 4,
        init_method: str = "kaiming"
    ):
        # Initialize LoRA components first
        super().__init__(base_layer, r, alpha, dropout, init_method)
        
        self.quant_bits = quant_bits
        
        # Quantize the base layer
        self._quantize_base_layer()
    
    def _get_layer_dimensions(self, layer: nn.Module) -> tuple:
        """Get input and output dimensions for both Linear and Conv1D layers"""
        if self._is_conv1d_layer(layer):
            # Conv1D stores weight as (in_features, out_features)
            weight_shape = layer.weight.shape
            return weight_shape[0], weight_shape[1]  # (in_features, out_features)
        else:
            # Standard Linear layer
            return layer.in_features, layer.out_features
    
    def _quantize_base_layer(self) -> None:
        """Quantize the base layer weights"""
        # Check if this is a Conv1D layer - bitsandbytes doesn't support Conv1D quantization
        if self._is_conv1d_layer(self.base_layer):
            print("Warning: bitsandbytes quantization not supported for Conv1D layers, using simple quantization")
            self._simple_quantize()
            return
            
        try:
            # Try to use bitsandbytes if available (only for Linear layers)
            import bitsandbytes as bnb
            
            # Store original weight and bias data
            original_weight = self.base_layer.weight.data.clone()
            original_bias = self.base_layer.bias.data.clone() if self.base_layer.bias is not None else None
            original_device = original_weight.device
            
            # Store original weight shape for merge operations
            self._original_weight_shape = original_weight.shape
            
            # Get layer dimensions
            in_features, out_features = self._get_layer_dimensions(self.base_layer)
            
            if self.quant_bits == 8:
                # 8-bit quantization
                quantized_layer = bnb.nn.Linear8bitLt(
                    in_features,
                    out_features,
                    bias=self.base_layer.bias is not None,
                    has_fp16_weights=False,
                    device=original_device
                )
            elif self.quant_bits == 4:
                # 4-bit quantization
                # Use nf4 for CPU compatibility, fp4 for GPU
                quant_type = "nf4" if original_device.type == "cpu" else "fp4"
                quantized_layer = bnb.nn.Linear4bit(
                    in_features,
                    out_features,
                    bias=self.base_layer.bias is not None,
                    compute_dtype=torch.float16,
                    compress_statistics=True,
                    quant_type=quant_type,
                    device=original_device
                )
            else:
                raise ValueError(f"Unsupported quantization bits: {self.quant_bits}")
            
            # Transfer weights to the quantized layer
            with torch.no_grad():
                quantized_layer.weight.data.copy_(original_weight)
                if original_bias is not None:
                    quantized_layer.bias.data.copy_(original_bias)
            
            # Move to correct device if needed
            quantized_layer = quantized_layer.to(original_device)
            
            # Replace the base layer
            self.base_layer = quantized_layer
                
        except ImportError:
            raise ImportError(
                "bitsandbytes is not installed. Install it with: pip install bitsandbytes"
            )
    
    def _simple_quantize(self) -> None:
        """
        Improved quantization using torch.ao.quantization APIs
        
        This method provides quantization implementation when bitsandbytes is not available.
        It uses the recommended torch.ao.quantization APIs for better accuracy and performance.
        
        For 8-bit: Uses quantize_dynamic, falls back to FakeQuantize
        For 4-bit: Uses FakeQuantize with 4-bit range
        """
        weight = self.base_layer.weight.data
        device = weight.device  # Store original device
        
        try:
            import torch.ao.quantization as ao_quant
            from torch.ao.quantization import HistogramObserver, FakeQuantize
            
            if self.quant_bits == 8:
                # Try torch.ao.quantization.quantize_dynamic first (recommended for 8-bit)
                try:
                    # Move to CPU for quantization to avoid device mismatch
                    weight_cpu = weight.cpu()
                    
                    # Create a temporary linear layer for quantization on CPU
                    in_features, out_features = self._get_layer_dimensions(self.base_layer)
                    temp_layer = torch.nn.Linear(
                        in_features,
                        out_features,
                        bias=self.base_layer.bias is not None,
                        device='cpu'
                    )
                    temp_layer.weight.data = weight_cpu.clone()
                    if self.base_layer.bias is not None:
                        temp_layer.bias.data = self.base_layer.bias.data.clone().cpu()
                    
                    # Apply dynamic quantization
                    quantized_layer = ao_quant.quantize_dynamic(
                        temp_layer,
                        {torch.nn.Linear},
                        dtype=torch.qint8
                    )
                    
                    # Extract quantized weights back and move to original device
                    quantized_weight = quantized_layer.weight().dequantize().to(device)
                    self.base_layer.weight.data = quantized_weight
                    if self.base_layer.bias is not None:
                        self.base_layer.bias.data = quantized_layer.bias().to(device)
                    
                    return
                    
                except Exception:
                    # Fall back to FakeQuantize for 8-bit on CPU
                    weight_cpu = weight.cpu()
                    fake_quant = FakeQuantize.with_args(
                        observer=HistogramObserver,
                        quant_min=-128,
                        quant_max=127,
                        dtype=torch.qint8,
                        qscheme=torch.per_tensor_symmetric,
                        reduce_range=False
                    )()
                    
                    quantized_weight = fake_quant(weight_cpu).to(device)
                    self.base_layer.weight.data = quantized_weight
                    
            elif self.quant_bits == 4:
                # Use FakeQuantize with 4-bit range on CPU
                weight_cpu = weight.cpu()
                fake_quant = FakeQuantize.with_args(
                    observer=HistogramObserver,
                    quant_min=-8,
                    quant_max=7,
                    dtype=torch.qint8,  # Use qint8 but with 4-bit range
                    qscheme=torch.per_tensor_symmetric,
                    reduce_range=False
                )()
                
                quantized_weight = fake_quant(weight_cpu).to(device)
                self.base_layer.weight.data = quantized_weight
                
            else:
                raise ValueError(f"Unsupported quantization bits: {self.quant_bits}")
                
        except Exception as e:
            print(f"Warning: torch.ao.quantization failed ({e}), using simple fallback")
            # Simple fallback quantization (keep on original device)
            if self.quant_bits == 8:
                scale = weight.abs().max() / 127
                quantized = torch.round(weight / scale).clamp(-128, 127)
                self.base_layer.weight.data = quantized * scale
            elif self.quant_bits == 4:
                scale = weight.abs().max() / 7
                quantized = torch.round(weight / scale).clamp(-8, 7)
                self.base_layer.weight.data = quantized * scale
    
    def merge(self) -> None:
        """Merge LoRA weights into quantized base layer"""
        if self._merged:
            return
        
        try:
            # Handle bitsandbytes quantized layers
            if hasattr(self.base_layer, 'weight') and hasattr(self.base_layer.weight, 'dequantize'):
                # Dequantize, add delta, and re-quantize
                dequantized_weight = self.base_layer.weight.dequantize()
                delta_weight = self.scaling * (self.B @ self.A)
                
                # Handle Conv1D vs Linear layer differences
                if self._is_conv1d_layer(self.base_layer):
                    # Conv1D has weights transposed compared to Linear
                    delta_weight = delta_weight.T
                
                # Get original weight shape before quantization
                target_shape = getattr(self, '_original_weight_shape', delta_weight.shape)
                
                # Ensure all tensors are on the same device
                device = dequantized_weight.device
                delta_weight = delta_weight.to(device)
                
                # Fix bitsandbytes dequantization shape issue
                if dequantized_weight.shape != target_shape:
                    # bitsandbytes compresses weights - we need to restore original shape
                    if dequantized_weight.numel() == target_shape.numel():
                        # Simple reshape if elements match
                        dequantized_weight = dequantized_weight.view(target_shape)
                    else:
                        # For 4-bit quantization, bitsandbytes may store compressed data
                        # Try to use the quantization state to get proper dequantization
                        try:
                            if hasattr(self.base_layer, 'quant_state') and self.base_layer.quant_state is not None:
                                import bitsandbytes.functional as F
                                # Use the quant_state to properly dequantize
                                dequantized_weight = F.dequantize_4bit(
                                    self.base_layer.weight.data,
                                    self.base_layer.quant_state,
                                    quant_type=getattr(self.base_layer.quant_state, 'quant_type', 'fp4')
                                )
                                # Reshape to target if needed
                                if dequantized_weight.shape != target_shape:
                                    dequantized_weight = dequantized_weight.view(target_shape)
                            else:
                                print(f"Warning: Cannot restore quantized shape {dequantized_weight.shape} to {target_shape}")
                                print("Skipping merge for this quantized layer")
                                self._merged = True
                                return
                        except Exception as inner_e:
                            print(f"Warning: Advanced dequantization failed ({inner_e})")
                            print("Skipping merge for this quantized layer")
                            self._merged = True
                            return
                
                # Ensure final shapes match for addition
                if dequantized_weight.shape != delta_weight.shape:
                    if dequantized_weight.numel() == delta_weight.numel():
                        dequantized_weight = dequantized_weight.view(delta_weight.shape)
                    else:
                        print(f"Warning: Shape mismatch after dequantization: {dequantized_weight.shape} vs {delta_weight.shape}")
                        print("Skipping merge for this quantized layer")
                        self._merged = True
                        return
                    
                # Perform the merge
                merged_weight = dequantized_weight + delta_weight
                
                # Note: This will lose quantization - the layer becomes unquantized after merge
                print("Warning: Merge operation dequantizes the layer (loses quantization benefits)")
                
                # Replace the quantized layer with a regular linear layer containing merged weights
                in_features, out_features = self._get_layer_dimensions(self.base_layer)
                new_layer = torch.nn.Linear(
                    in_features, 
                    out_features, 
                    bias=self.base_layer.bias is not None,
                    device=device,
                    dtype=merged_weight.dtype
                )
                new_layer.weight.data = merged_weight
                if self.base_layer.bias is not None:
                    new_layer.bias.data = self.base_layer.bias.data.clone()
                
                # Replace the quantized layer
                self.base_layer = new_layer
            else:
                # Standard merge for simple quantization with Conv1D handling
                delta_weight = self.scaling * (self.B @ self.A)
                
                # Handle Conv1D vs Linear layer differences
                if self._is_conv1d_layer(self.base_layer):
                    # Conv1D has weights transposed compared to Linear
                    delta_weight = delta_weight.T
                
                self.base_layer.weight.data += delta_weight
                
            self._merged = True
                
        except Exception as e:
            print(f"Warning: Could not merge quantized weights: {e}")
            # Fallback to standard merge
            super().merge()
    
    def extra_repr(self) -> str:
        base_repr = super().extra_repr()
        return f"{base_repr}, quant_bits={self.quant_bits}"


def create_adapter(
    layer: nn.Module,
    adapter_type: str = "lora",
    **kwargs
) -> AdapterBase:
    """
    Factory function to create adapters
    
    Args:
        layer: Base layer to adapt
        adapter_type: Type of adapter ("lora", "qlora")
        **kwargs: Additional arguments for adapter creation
        
    Returns:
        Created adapter instance
    """
    if adapter_type.lower() == "lora":
        return LoRAAdapter(layer, **kwargs)
    elif adapter_type.lower() == "qlora":
        return QLoRAAdapter(layer, **kwargs)
    else:
        raise ValueError(f"Unknown adapter type: {adapter_type}")


def get_adapter_info(adapter: AdapterBase) -> Dict[str, Any]:
    """
    Get information about an adapter
    
    Args:
        adapter: The adapter to inspect
        
    Returns:
        Dictionary with adapter information
    """
    info = {
        "type": adapter.__class__.__name__,
        "is_merged": adapter.is_merged,
        "trainable_params": sum(p.numel() for p in adapter.parameters() if p.requires_grad),
        "total_params": sum(p.numel() for p in adapter.parameters())
    }
    
    if isinstance(adapter, (LoRAAdapter, QLoRAAdapter)):
        info.update({
            "r": adapter.r,
            "rank": adapter.r,  # Keep both for compatibility
            "alpha": adapter.alpha,
            "scaling": adapter.scaling
        })
        
    if isinstance(adapter, QLoRAAdapter):
        info["quantization_bits"] = adapter.quant_bits
    
    return info 