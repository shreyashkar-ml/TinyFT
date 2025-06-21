import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import (
    LinearLR, CosineAnnealingLR, CosineAnnealingWarmRestarts, 
    PolynomialLR, ConstantLR
)
from typing import Dict, List, Optional, Any, Union, Callable
import logging
import time
import math
from pathlib import Path
from tqdm import tqdm

from .utils import (
    setup_logging, create_progress_format, save_checkpoint, 
    load_checkpoint, get_device, MemoryTracker, format_time, format_number
)
from .datasets import SFTDataset, CPTDataset, DatasetBuilder

logger = logging.getLogger(__name__)


class TinyFTTrainer:
    """
    Unified trainer for supervised fine-tuning and continued pre-training
    
    Supports LoRA, QLoRA, and full fine-tuning with comprehensive logging,
    progress tracking, and memory optimization features.
    """
    
    def __init__(
        self,
        model: nn.Module,
        training_type: str = "sft",
        dataset: Optional[Union[SFTDataset, CPTDataset]] = None,
        eval_dataset: Optional[Union[SFTDataset, CPTDataset]] = None,
        tokenizer = None,
        
        # Training hyperparameters
        learning_rate: float = 2e-4,
        batch_size: int = 8,
        gradient_accumulation_steps: int = 4,
        num_epochs: int = 3,
        max_steps: Optional[int] = None,
        warmup_steps: int = 100,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
        
        # Optimization settings
        optimizer_type: str = "adamw",
        lr_scheduler_type: str = "cosine",
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        adam_epsilon: float = 1e-8,
        
        # Mixed precision
        fp16: bool = True,
        bf16: bool = False,
        
        # Logging and monitoring
        logging_backend: str = "tensorboard",
        logging_steps: int = 10,
        eval_steps: int = 500,
        save_steps: int = 1000,
        
        # Paths
        output_dir: str = "./outputs",
        log_dir: str = "./logs",
        run_name: Optional[str] = None,
        
        # Other settings
        seed: int = 42,
        dataloader_num_workers: int = 4,
        gradient_checkpointing: bool = False,
        **kwargs
    ):
        """
        Initialize the trainer
        
        Args:
            model: Model to train
            training_type: Type of training ("sft" or "cpt")
            dataset: Training dataset
            eval_dataset: Evaluation dataset (optional)
            tokenizer: Tokenizer for the model
            **kwargs: Additional training arguments
        """
        self.model = model
        self.training_type = training_type.lower()
        self.dataset = dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        
        # Training settings
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.num_epochs = num_epochs
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        
        # Optimization
        self.optimizer_type = optimizer_type
        self.lr_scheduler_type = lr_scheduler_type
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.adam_epsilon = adam_epsilon
        
        # Mixed precision
        self.fp16 = fp16
        self.bf16 = bf16
        
        # Logging
        self.logging_backend = logging_backend
        self.logging_steps = logging_steps
        self.eval_steps = eval_steps
        self.save_steps = save_steps
        
        # Paths
        self.output_dir = Path(output_dir)
        self.log_dir = Path(log_dir)
        self.run_name = run_name
        
        # Other
        self.seed = seed
        self.dataloader_num_workers = dataloader_num_workers
        self.gradient_checkpointing = gradient_checkpointing
        
        # Initialize training components
        self.device = get_device(prefer_cuda=True)
        self.model = self.model.to(self.device)
        
        # Setup logging
        self.loggers = setup_logging(
            backend=self.logging_backend,
            project_name=f"tinyft_{self.training_type}",
            run_name=self.run_name,
            log_dir=str(self.log_dir)
        )
        
        # Initialize training state
        self.global_step = 0
        self.epoch = 0
        self.best_eval_loss = float('inf')
        self.memory_tracker = MemoryTracker(self.device)
        
        # Setup mixed precision
        self.scaler = None
        if self.fp16:
            self.scaler = torch.amp.GradScaler('cuda')
        
        # Enable gradient checkpointing if requested
        if self.gradient_checkpointing and hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
        
        logger.info(f"Initialized TinyFT Trainer for {self.training_type.upper()}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Mixed precision: FP16={self.fp16}, BF16={self.bf16}")
    
    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """Setup optimizer"""
        # Get trainable parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        if self.optimizer_type.lower() == "adamw":
            optimizer = AdamW(
                trainable_params,
                lr=self.learning_rate,
                betas=(self.adam_beta1, self.adam_beta2),
                eps=self.adam_epsilon,
                weight_decay=self.weight_decay
            )
        elif self.optimizer_type.lower() == "sgd":
            optimizer = SGD(
                trainable_params,
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_type}")
        
        logger.info(f"Setup {self.optimizer_type.upper()} optimizer with {len(trainable_params)} trainable parameters")
        return optimizer
    
    def _setup_scheduler(self, optimizer: torch.optim.Optimizer, num_training_steps: int):
        """Setup learning rate scheduler"""
        if self.lr_scheduler_type.lower() == "linear":
            scheduler = LinearLR(
                optimizer,
                start_factor=0.1,
                total_iters=self.warmup_steps
            )
        elif self.lr_scheduler_type.lower() == "cosine":
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=num_training_steps - self.warmup_steps,
                eta_min=self.learning_rate * 0.1
            )
        elif self.lr_scheduler_type.lower() == "cosine_with_restarts":
            scheduler = CosineAnnealingWarmRestarts(
                optimizer,
                T_0=num_training_steps // 4,
                T_mult=2
            )
        elif self.lr_scheduler_type.lower() == "constant":
            scheduler = ConstantLR(optimizer, factor=1.0)
        else:
            raise ValueError(f"Unsupported scheduler: {self.lr_scheduler_type}")
        
        return scheduler
    
    def _setup_dataloaders(self) -> tuple:
        """Setup training and evaluation dataloaders"""
        train_dataloader = None
        eval_dataloader = None
        
        if self.dataset:
            train_dataloader = DatasetBuilder.create_dataloader(
                self.dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.dataloader_num_workers,
                pin_memory=True
            )
        
        if self.eval_dataset:
            eval_dataloader = DatasetBuilder.create_dataloader(
                self.eval_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.dataloader_num_workers,
                pin_memory=True
            )
        
        return train_dataloader, eval_dataloader
    
    def _compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute loss based on training type"""
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)
        
        # Forward pass
        if self.fp16:
            with torch.amp.autocast(device_type=self.device.type):
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss
        else:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
        
        return loss
    
    def _training_step(
        self, 
        batch: Dict[str, torch.Tensor], 
        optimizer: torch.optim.Optimizer
    ) -> float:
        """Perform a single training step"""
        loss = self._compute_loss(batch)
        
        # Scale loss for gradient accumulation
        loss = loss / self.gradient_accumulation_steps
        
        # Backward pass
        if self.fp16 and self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        return loss.item() * self.gradient_accumulation_steps
    
    def _optimizer_step(self, optimizer: torch.optim.Optimizer, scheduler):
        """Perform optimizer step with gradient clipping"""
        if self.fp16 and self.scaler:
            # Unscale gradients for clipping
            self.scaler.unscale_(optimizer)
            
            # Clip gradients
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            
            # Optimizer step
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            # Clip gradients
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            
            # Optimizer step
            optimizer.step()
        
        # Scheduler step
        scheduler.step()
        
        # Zero gradients
        optimizer.zero_grad()
    
    def _evaluate(self, eval_dataloader: DataLoader) -> Dict[str, float]:
        """Run evaluation"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in eval_dataloader:
                loss = self._compute_loss(batch)
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        perplexity = math.exp(avg_loss) if avg_loss < 10 else float('inf')
        
        self.model.train()
        return {"eval_loss": avg_loss, "eval_perplexity": perplexity}
    
    def _log_metrics(self, metrics: Dict[str, Any], step: int):
        """Log metrics to configured backends"""
        # TensorBoard logging
        if "tensorboard" in self.loggers:
            tb_logger = self.loggers["tensorboard"]
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    tb_logger.add_scalar(key, value, step)
        
        # Weights & Biases logging
        if "wandb" in self.loggers:
            wandb = self.loggers["wandb"]
            wandb.log(metrics, step=step)
    
    def train(self) -> Dict[str, Any]:
        """
        Main training loop
        
        Returns:
            Dictionary with training statistics
        """
        if not self.dataset:
            raise ValueError("No training dataset provided")
        
        # Setup training components
        train_dataloader, eval_dataloader = self._setup_dataloaders()
        optimizer = self._setup_optimizer()
        self._optimizer = optimizer  # Store for checkpoint saving
        
        # Calculate training steps
        steps_per_epoch = max(1, len(train_dataloader) // self.gradient_accumulation_steps)
        if self.max_steps:
            num_training_steps = self.max_steps
            self.num_epochs = math.ceil(self.max_steps / steps_per_epoch)
        else:
            num_training_steps = steps_per_epoch * self.num_epochs
        
        scheduler = self._setup_scheduler(optimizer, num_training_steps)
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Training statistics
        training_stats = {
            "total_steps": num_training_steps,
            "steps_per_epoch": steps_per_epoch,
            "total_epochs": self.num_epochs,
            "start_time": time.time()
        }
        
        logger.info(f"Starting training for {self.num_epochs} epochs ({num_training_steps} steps)")
        logger.info(f"Batch size: {self.batch_size}, Gradient accumulation: {self.gradient_accumulation_steps}")
        logger.info(f"Effective batch size: {self.batch_size * self.gradient_accumulation_steps}")
        
        # Training loop
        self.model.train()
        accumulated_loss = 0.0
        step_start_time = time.time()
        
        # Progress bar for epochs
        epoch_pbar = tqdm(range(self.num_epochs), desc="Epochs", position=0)
        
        for epoch in epoch_pbar:
            self.epoch = epoch
            epoch_loss = 0.0
            epoch_steps = 0
            
            # Progress bar for steps within epoch
            step_pbar = tqdm(
                train_dataloader, 
                desc=f"Epoch {epoch+1}/{self.num_epochs}", 
                position=1,
                leave=False
            )
            
            for step, batch in enumerate(step_pbar):
                # Training step
                step_loss = self._training_step(batch, optimizer)
                accumulated_loss += step_loss
                epoch_loss += step_loss
                epoch_steps += 1
                
                # Gradient accumulation
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    self._optimizer_step(optimizer, scheduler)
                    self.global_step += 1
                    
                    # Calculate metrics
                    avg_loss = accumulated_loss / self.gradient_accumulation_steps
                    current_lr = scheduler.get_last_lr()[0]
                    
                    # Calculate speed
                    step_time = time.time() - step_start_time
                    it_per_sec = self.gradient_accumulation_steps / step_time if step_time > 0 else 0
                    
                    # Update memory tracker
                    current_memory = self.memory_tracker.update()
                    
                    # Create progress string
                    examples_trained = self.global_step * self.batch_size * self.gradient_accumulation_steps
                    total_examples = len(self.dataset)
                    progress_str = create_progress_format(
                        examples_trained=examples_trained,
                        total_examples=total_examples,
                        epoch=epoch + 1,
                        total_epochs=self.num_epochs,
                        it_per_sec=it_per_sec,
                        loss=avg_loss,
                        lr=current_lr
                    )
                    
                    # Update progress bars
                    step_pbar.set_description(progress_str)
                    epoch_pbar.set_postfix({
                        "Loss": f"{avg_loss:.4f}",
                        "LR": f"{current_lr:.2e}",
                        "Mem": f"{current_memory:.0f}MB"
                    })
                    
                    # Logging
                    if self.global_step % self.logging_steps == 0:
                        metrics = {
                            "train_loss": avg_loss,
                            "learning_rate": current_lr,
                            "epoch": epoch + 1,
                            "global_step": self.global_step,
                            "memory_mb": current_memory,
                            "it_per_sec": it_per_sec
                        }
                        self._log_metrics(metrics, self.global_step)
                    
                    # Evaluation
                    if eval_dataloader and self.eval_steps and self.global_step % self.eval_steps == 0:
                        eval_metrics = self._evaluate(eval_dataloader)
                        eval_metrics["global_step"] = self.global_step
                        self._log_metrics(eval_metrics, self.global_step)
                        
                        # Save best model
                        if eval_metrics["eval_loss"] < self.best_eval_loss:
                            self.best_eval_loss = eval_metrics["eval_loss"]
                            self.save_checkpoint("best_model")
                    
                    # Save checkpoint
                    if self.save_steps and self.global_step % self.save_steps == 0:
                        self.save_checkpoint(f"checkpoint-{self.global_step}")
                    
                    # Reset for next accumulation
                    accumulated_loss = 0.0
                    step_start_time = time.time()
                    
                    # Check if max steps reached
                    if self.max_steps and self.global_step >= self.max_steps:
                        break
            
            # End of epoch
            avg_epoch_loss = epoch_loss / epoch_steps if epoch_steps > 0 else 0.0
            logger.info(f"Epoch {epoch+1} completed. Average loss: {avg_epoch_loss:.4f}")
            
            if self.max_steps and self.global_step >= self.max_steps:
                break
        
        # Training completed
        total_time = time.time() - training_stats["start_time"]
        training_stats.update({
            "final_loss": avg_epoch_loss,
            "total_time": total_time,
            "final_step": self.global_step,
            "final_epoch": self.epoch + 1
        })
        
        logger.info(f"Training completed in {format_time(total_time)}")
        logger.info(f"Final loss: {training_stats.get('final_loss', 0):.4f}")
        
        # Save final model
        self.save_checkpoint("final_model")
        
        return training_stats
    
    def save_checkpoint(self, checkpoint_name: str):
        """Save training checkpoint"""
        checkpoint_path = self.output_dir / f"{checkpoint_name}.pt"
        
        # Create metadata
        metadata = {
            "training_type": self.training_type,
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_eval_loss": self.best_eval_loss,
            "model_class": self.model.__class__.__name__
        }
        
        # Save checkpoint
        save_checkpoint(
            model=self.model,
            optimizer=getattr(self, '_optimizer', None),  # Save optimizer if available
            epoch=self.epoch,
            step=self.global_step,
            loss=0.0,  # We'll track this properly later
            save_path=checkpoint_path,
            metadata=metadata
        )
        
        logger.info(f"Saved checkpoint: {checkpoint_path}")
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get current training statistics"""
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        
        return {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "training_type": self.training_type,
            "trainable_parameters": trainable_params,
            "total_parameters": total_params,
            "trainable_percentage": (trainable_params / total_params) * 100,
            "device": str(self.device),
            "memory_usage_mb": self.memory_tracker.peak_memory
        } 