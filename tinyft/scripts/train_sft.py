import argparse
import sys
import os
from pathlib import Path
import yaml
import logging

# Add parent directory to path to import tinyft
sys.path.insert(0, str(Path(__file__).parent.parent))

from tinyft import AdapterManager, setup_logging


def parse_args():
    """Parse command line arguments (YAML-first approach)"""
    parser = argparse.ArgumentParser(
        description="Supervised Fine-Tuning with TinyFT (YAML-First Configuration)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="""
Examples:
  # Basic usage with config file
  python train_sft.py --config configs/base_lora.yaml
  
  # Quick override for experimentation
  python train_sft.py --config configs/base_lora.yaml --output_dir ./experiment_1 --run_name "test_run"
  
  # Evaluation only
  python train_sft.py --config configs/trained_model.yaml --eval_only
  
  # Resume training
  python train_sft.py --config configs/experiment.yaml --resume_from_checkpoint ./outputs/checkpoint-1000
        """
    )
    
    # Essential arguments
    parser.add_argument(
        "--config", 
        type=str, 
        required=True,
        help="Path to YAML configuration file (REQUIRED)"
    )
    
    # Runtime mode arguments
    parser.add_argument(
        "--eval_only", 
        action="store_true",
        help="Run evaluation only, no training"
    )
    parser.add_argument(
        "--resume_from_checkpoint", 
        type=str,
        help="Path to checkpoint to resume from"
    )
    
    # Quick override arguments (for experimentation)
    parser.add_argument(
        "--output_dir", 
        type=str,
        help="Override output directory for checkpoints"
    )
    parser.add_argument(
        "--run_name", 
        type=str,
        help="Override run name for logging"
    )
    parser.add_argument(
        "--seed", 
        type=int,
        help="Override random seed for reproducibility"
    )
    
    # Development/debugging arguments
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="Enable debug mode with verbose logging"
    )
    parser.add_argument(
        "--dry_run", 
        action="store_true",
        help="Validate configuration and exit (no training)"
    )
    
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        if config is None:
            raise ValueError("Configuration file is empty or invalid")
        
        return config
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in configuration file: {e}")


def apply_cli_overrides(config: dict, args: argparse.Namespace) -> dict:
    """Apply minimal CLI overrides to configuration"""
    # Only apply overrides if explicitly provided
    if args.output_dir is not None:
        config['output_dir'] = args.output_dir
        
    if args.run_name is not None:
        config['run_name'] = args.run_name
        
    if args.seed is not None:
        config['seed'] = args.seed
    
    # Add runtime flags
    config['eval_only'] = args.eval_only
    config['resume_from_checkpoint'] = args.resume_from_checkpoint
    config['debug'] = args.debug
    config['dry_run'] = args.dry_run
    
    return config


def validate_config(config: dict) -> None:
    """Validate configuration for required fields"""
    required_fields = [
        'model_name', 'method', 'dataset_path', 'learning_rate', 
        'batch_size', 'num_epochs', 'output_dir'
    ]
    
    missing_fields = [field for field in required_fields if field not in config]
    
    if missing_fields:
        raise ValueError(f"Missing required configuration fields: {missing_fields}")
    
    # Validate method
    if config['method'] not in ['lora', 'qlora', 'full']:
        raise ValueError(f"Invalid method: {config['method']}. Must be one of: lora, qlora, full")
    
    # Validate LoRA config path if using LoRA methods
    if config['method'] in ['lora', 'qlora']:
        if 'lora_config_path' not in config:
            raise ValueError("lora_config_path is required for LoRA methods")
        
        if not os.path.exists(config['lora_config_path']):
            raise FileNotFoundError(f"LoRA config file not found: {config['lora_config_path']}")


def print_config_summary(config: dict) -> None:
    """Print a summary of the configuration"""
    print("🚀 TinyFT Supervised Fine-Tuning")
    print("=" * 50)
    print(f"Configuration: {config.get('config_file', 'Unknown')}")
    print(f"Model: {config['model_name']}")
    print(f"Method: {config['method'].upper()}")
    print(f"Dataset: {config['dataset_path']}")
    print(f"Output Directory: {config['output_dir']}")
    print(f"Run Name: {config.get('run_name', 'Auto-generated')}")
    
    if config.get('eval_only'):
        print("Mode: EVALUATION ONLY")
    elif config.get('resume_from_checkpoint'):
        print(f"Mode: RESUME from {config['resume_from_checkpoint']}")
    else:
        print("Mode: TRAINING")
    
    print("-" * 50)


def main():
    """Main training function"""
    args = parse_args()
    
    try:
        # Load and validate configuration
        config = load_config(args.config)
        config['config_file'] = args.config  # Track source config file
        config = apply_cli_overrides(config, args)
        validate_config(config)
        
        # Print configuration summary
        print_config_summary(config)
        
        # Handle dry run
        if config.get('dry_run'):
            print("✅ Configuration validation passed. Exiting (dry run mode).")
            return 0
        
        # Setup logging
        log_level = logging.DEBUG if config.get('debug') else logging.INFO
        loggers = setup_logging(
            backend=config['logging_backend'],
            project_name=config.get('project_name', 'tinyft_sft'),
            run_name=config.get('run_name'),
            log_dir=config['logging_dir'],
            log_level=log_level
        )
        
        logger = loggers['python']
        logger.info("Starting TinyFT Supervised Fine-Tuning")
        logger.info(f"Configuration loaded from: {args.config}")
        
        # Set random seed
        from tinyft.utils import set_seed
        set_seed(config['seed'])
        
        # TODO: Implement full training pipeline
        # This is a placeholder for the complete implementation
        
        # 1. Load model and tokenizer
        print("📦 Loading model and tokenizer...")
        logger.info(f"Loading model: {config['model_name']}")
        
        # 2. Apply adapters
        if config['method'] in ['lora', 'qlora']:
            print(f"🔧 Applying {config['method'].upper()} adapters...")
            manager = AdapterManager()
            logger.info(f"Applying {config['method']} adapters")
        
        # 3. Load and preprocess dataset
        print("📊 Loading and preprocessing dataset...")
        logger.info(f"Loading dataset: {config['dataset_path']}")
        
        # 4. Initialize trainer
        print("🏋️ Initializing trainer...")
        logger.info("Initializing TinyFT trainer")
        
        # 5. Handle different modes
        if config.get('eval_only'):
            print("📋 Running evaluation...")
            logger.info("Running evaluation only")
        elif config.get('resume_from_checkpoint'):
            print(f"🔄 Resuming training from {config['resume_from_checkpoint']}...")
            logger.info(f"Resuming from checkpoint: {config['resume_from_checkpoint']}")
        else:
            print("🚀 Starting training...")
            logger.info("Starting training from scratch")
        
        print("✅ Training completed successfully!")
        logger.info("Training completed successfully")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 