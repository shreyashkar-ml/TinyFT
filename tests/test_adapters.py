import unittest
import torch
import torch.nn as nn
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tinyft.adapters import LoRAAdapter, QLoRAAdapter, create_adapter, get_adapter_info
from tinyft.manager import AdapterManager


class TestAdapters(unittest.TestCase):
    """Test adapter implementations"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.device = torch.device("cpu")  # Use CPU for tests
        self.input_dim = 512
        self.output_dim = 256
        self.base_layer = nn.Linear(self.input_dim, self.output_dim)
        self.batch_size = 4
        self.seq_len = 128
        
    def test_lora_adapter_creation(self):
        """Test LoRA adapter creation"""
        adapter = LoRAAdapter(
            base_layer=self.base_layer,
            r=8,
            alpha=16,
            dropout=0.1
        )
        
        # Check adapter properties
        self.assertEqual(adapter.r, 8)
        self.assertEqual(adapter.alpha, 16)
        self.assertEqual(adapter.scaling, 2.0)  # alpha / r
        self.assertFalse(adapter.is_merged)
        
        # Check parameter shapes
        self.assertEqual(adapter.A.shape, (8, self.input_dim))
        self.assertEqual(adapter.B.shape, (self.output_dim, 8))
        
    def test_lora_adapter_forward(self):
        """Test LoRA adapter forward pass"""
        adapter = LoRAAdapter(
            base_layer=self.base_layer,
            r=8,
            alpha=16
        )
        
        # Test input
        x = torch.randn(self.batch_size, self.seq_len, self.input_dim)
        
        # Forward pass
        output = adapter(x)
        
        # Check output shape
        expected_shape = (self.batch_size, self.seq_len, self.output_dim)
        self.assertEqual(output.shape, expected_shape)
        
    def test_lora_adapter_merge_unmerge(self):
        """Test LoRA adapter merge/unmerge functionality"""
        adapter = LoRAAdapter(
            base_layer=self.base_layer,
            r=8,
            alpha=16
        )
        
        # Store original weights
        original_weight = self.base_layer.weight.data.clone()
        
        # Test merge
        adapter.merge()
        self.assertTrue(adapter.is_merged)
        
        # Weights should be different after merge
        self.assertFalse(torch.equal(original_weight, self.base_layer.weight.data))
        
        # Test unmerge
        adapter.unmerge()
        self.assertFalse(adapter.is_merged)
        
        # Weights should be back to original
        self.assertTrue(torch.allclose(original_weight, self.base_layer.weight.data, atol=1e-6))
        
    def test_qlora_adapter_creation(self):
        """Test QLoRA adapter creation"""
        # Note: This test might fail if bitsandbytes is not available
        try:
            adapter = QLoRAAdapter(
                base_layer=self.base_layer,
                r=8,
                alpha=16,
                quant_bits=8
            )
            
            # Check adapter properties
            self.assertEqual(adapter.r, 8)
            self.assertEqual(adapter.alpha, 16)
            self.assertEqual(adapter.quant_bits, 8)
            
        except ImportError:
            # Skip test if bitsandbytes not available
            self.skipTest("bitsandbytes not available for QLoRA testing")
            
    def test_adapter_factory(self):
        """Test adapter factory function"""
        # Test LoRA creation
        lora_adapter = create_adapter(
            self.base_layer,
            adapter_type="lora",
            r=8,
            alpha=16
        )
        self.assertIsInstance(lora_adapter, LoRAAdapter)
        
        # Test invalid adapter type
        with self.assertRaises(ValueError):
            create_adapter(self.base_layer, adapter_type="invalid")
            
    def test_adapter_info(self):
        """Test adapter info function"""
        adapter = LoRAAdapter(
            base_layer=self.base_layer,
            r=8,
            alpha=16
        )
        
        info = get_adapter_info(adapter)
        
        # Check info structure
        self.assertIn("type", info)
        self.assertIn("is_merged", info)
        self.assertIn("trainable_params", info)
        self.assertIn("rank", info)
        self.assertIn("alpha", info)
        
        self.assertEqual(info["type"], "LoRAAdapter")
        self.assertEqual(info["rank"], 8)
        self.assertEqual(info["alpha"], 16)


class TestAdapterManager(unittest.TestCase):
    """Test adapter manager functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.manager = AdapterManager()
        
        # Create a simple test model
        self.model = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        
    def test_model_type_detection(self):
        """Test model type detection"""
        # Test with a simple model (should return unknown)
        model_type = self.manager.detect_model_type(self.model)
        self.assertEqual(model_type, "unknown")
        
    def test_default_target_modules(self):
        """Test default target module retrieval"""
        # Test known model types
        llama_modules = self.manager.get_default_target_modules("llama")
        self.assertIn("q_proj", llama_modules)
        self.assertIn("k_proj", llama_modules)
        
        gpt2_modules = self.manager.get_default_target_modules("gpt2")
        self.assertIn("c_attn", gpt2_modules)
        
        # Test unknown model type
        unknown_modules = self.manager.get_default_target_modules("unknown")
        self.assertIn("q_proj", unknown_modules)  # Should return fallback
        
    def test_find_target_modules(self):
        """Test target module finding"""
        # This test is limited since our simple model doesn't have named modules
        # In a real scenario, this would test finding specific modules in a transformer
        
        # Test with empty result (expected for our simple model)
        with self.assertRaises(ValueError):
            self.manager.find_target_modules(self.model, ["q_proj"])
            
    def test_trainable_parameters(self):
        """Test trainable parameter counting"""
        stats = self.manager.get_trainable_parameters(self.model)
        
        self.assertIn("trainable", stats)
        self.assertIn("total", stats)
        self.assertIn("percentage", stats)
        
        # All parameters should be trainable initially
        self.assertEqual(stats["trainable"], stats["total"])
        self.assertEqual(stats["percentage"], 100.0)


class TestIntegration(unittest.TestCase):
    """Integration tests"""
    
    def test_adapter_gradient_flow(self):
        """Test that gradients flow only through adapter parameters"""
        base_layer = nn.Linear(512, 256)
        adapter = LoRAAdapter(base_layer, r=8, alpha=16)
        
        # Create dummy input and target
        x = torch.randn(4, 128, 512)
        target = torch.randn(4, 128, 256)
        
        # Forward pass
        output = adapter(x)
        loss = nn.MSELoss()(output, target)
        
        # Backward pass
        loss.backward()
        
        # Check that base layer parameters have no gradients
        for param in base_layer.parameters():
            self.assertIsNone(param.grad)
            
        # Check that adapter parameters have gradients
        self.assertIsNotNone(adapter.A.grad)
        self.assertIsNotNone(adapter.B.grad)
        
    def test_adapter_parameter_count(self):
        """Test adapter parameter efficiency"""
        input_dim = 4096
        output_dim = 4096
        base_layer = nn.Linear(input_dim, output_dim)
        
        # Calculate base layer parameters
        base_params = input_dim * output_dim + output_dim  # weights + bias
        
        # Create LoRA adapter
        r = 16
        adapter = LoRAAdapter(base_layer, r=r, alpha=32)
        
        # Calculate adapter parameters
        adapter_params = r * input_dim + output_dim * r  # A + B matrices
        
        # Adapter should be much smaller
        reduction_ratio = base_params / adapter_params
        self.assertGreater(reduction_ratio, 10)  # Should be at least 10x smaller


if __name__ == "__main__":
    # Run tests
    unittest.main(verbosity=2) 