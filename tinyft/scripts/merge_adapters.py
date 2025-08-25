import argparse
import sys
import os
from pathlib import Path
import torch
import logging

# Add parent directory to path to import tinyft
sys.path.insert(0, str(Path(__file__).parent.parent))

from tinyft import AdapterManager, setup_logging


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Merge LoRA/QLoRA adapters into base model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model arguments
    parser.add_argument(
        "--base_model", 
        type=str, 
        required=True,
        help="Base model name or path"
    )
    parser.add_argument(
        "--adapter_path", 
        type=str, 
        required=True,
        help="Path to adapter checkpoint"
    )
    parser.add_argument(
        "--output_path", 
        type=str, 
        required=True,
        help="Output path for merged model"
    )
    
    # Adapter configuration
    parser.add_argument(
        "--adapter_method", 
        type=str, 
        choices=["lora", "qlora"],
        default="lora",
        help="Type of adapter to merge"
    )
    parser.add_argument(
        "--target_modules", 
        type=str, 
        nargs="+",
        help="Target modules (auto-detect if not specified)"
    )
    
    # LoRA parameters (needed for reconstruction)
    parser.add_argument(
        "--lora_r", 
        type=int, 
        default=16,
        help="LoRA rank"
    )
    parser.add_argument(
        "--lora_alpha", 
        type=int, 
        default=32,
        help="LoRA alpha scaling factor"
    )
    parser.add_argument(
        "--lora_dropout", 
        type=float, 
        default=0.1,
        help="LoRA dropout"
    )
    
    # Output options
    parser.add_argument(
        "--save_tokenizer", 
        action="store_true",
        help="Also save tokenizer with merged model"
    )
    parser.add_argument(
        "--push_to_hub", 
        type=str,
        help="Push merged model to HuggingFace Hub (provide repo name)"
    )
    parser.add_argument(
        "--private", 
        action="store_true",
        help="Make HuggingFace repo private"
    )
    
    # Other options
    parser.add_argument(
        "--device", 
        type=str, 
        default="auto",
        help="Device to use for merging (auto, cpu, cuda)"
    )
    parser.add_argument(
        "--max_memory", 
        type=str,
        help="Maximum memory per device (e.g., '8GB')"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()


def load_base_model_and_tokenizer(model_name: str, device: str = "auto"):
    """Load base model and tokenizer"""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        print(f"📦 Loading base model: {model_name}")
        
        # Determine device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map=device if device != "cpu" else None,
            low_cpu_mem_usage=True
        )
        
        print(f"✅ Loaded model on {device}")
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        raise


def apply_and_merge_adapters(
    model, 
    adapter_path: str, 
    adapter_method: str,
    target_modules=None,
    **adapter_kwargs
):
    """Apply adapters to model and merge them"""
    print(f"🔧 Applying {adapter_method.upper()} adapters...")
    
    # Initialize adapter manager
    manager = AdapterManager()
    
    # Apply adapters to model
    model = manager.apply_adapters(
        model=model,
        method=adapter_method,
        target_modules=target_modules or "auto",
        **adapter_kwargs
    )
    
    # Load adapter weights
    print(f"📥 Loading adapter weights from: {adapter_path}")
    manager.load_adapters(model, adapter_path)
    
    # Merge adapters into base weights
    print("🔀 Merging adapters into base model...")
    manager.merge_adapters(model)
    
    # Get statistics
    stats = manager.get_trainable_parameters(model)
    print(f"📊 Model statistics:")
    print(f"   - Total parameters: {stats['total']:,}")
    print(f"   - Trainable parameters: {stats['trainable']:,}")
    print(f"   - Adapters merged: {stats['adapters']}")
    
    return model


def save_merged_model(
    model, 
    tokenizer, 
    output_path: str, 
    save_tokenizer: bool = True,
    push_to_hub: str = None,
    private: bool = False
):
    """Save the merged model"""
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"💾 Saving merged model to: {output_path}")
    
    # Save model
    model.save_pretrained(
        output_path,
        safe_serialization=True,
        max_shard_size="5GB"
    )
    
    # Save tokenizer
    if save_tokenizer and tokenizer:
        tokenizer.save_pretrained(output_path)
        print("✅ Saved tokenizer")
    
    print("✅ Saved merged model")
    
    # Push to Hub if requested
    if push_to_hub:
        print(f"🚀 Pushing to HuggingFace Hub: {push_to_hub}")
        try:
            model.push_to_hub(
                push_to_hub,
                private=private,
                safe_serialization=True
            )
            if save_tokenizer and tokenizer:
                tokenizer.push_to_hub(push_to_hub, private=private)
            print("✅ Successfully pushed to Hub")
        except Exception as e:
            print(f"❌ Failed to push to Hub: {e}")


def main():
    """Main merging function"""
    args = parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    loggers = setup_logging(
        backend="tensorboard",
        project_name="tinyft_merge",
        log_dir="./logs",
        log_level=log_level
    )
    
    logger = loggers['python']
    logger.info("Starting TinyFT Adapter Merging")
    
    try:
        print("🔀 TinyFT Adapter Merging")
        print(f"Base Model: {args.base_model}")
        print(f"Adapter Path: {args.adapter_path}")
        print(f"Output Path: {args.output_path}")
        print(f"Method: {args.adapter_method.upper()}")
        print("-" * 50)
        
        # Validate paths
        adapter_path = Path(args.adapter_path)
        if not adapter_path.exists():
            raise FileNotFoundError(f"Adapter path not found: {adapter_path}")
        
        # Load base model and tokenizer
        model, tokenizer = load_base_model_and_tokenizer(
            args.base_model, 
            args.device
        )
        
        # Apply and merge adapters
        model = apply_and_merge_adapters(
            model=model,
            adapter_path=args.adapter_path,
            adapter_method=args.adapter_method,
            target_modules=args.target_modules,
            r=args.lora_r,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout
        )
        
        # Save merged model
        save_merged_model(
            model=model,
            tokenizer=tokenizer,
            output_path=args.output_path,
            save_tokenizer=args.save_tokenizer,
            push_to_hub=args.push_to_hub,
            private=args.private
        )
        
        print("✅ Adapter merging completed successfully!")
        print(f"📁 Merged model saved to: {args.output_path}")
        
        # Memory cleanup
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
    except Exception as e:
        logger.error(f"Merging failed: {e}")
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 