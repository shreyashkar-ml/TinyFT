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
    load_checkpoint, get_device, MemoryTracker, format_time, format_number,
    set_seed,
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


class TinyGRPOTrainer:
    """
    Minimal Group Relative Policy Optimization (GRPO) trainer compatible with TinyFT.

    This trainer implements a simple, dependency-light GRPO loop:
    - Samples multiple responses per prompt.
    - Computes per-group normalized advantages.
    - Applies a token-wise policy gradient objective over generated tokens.

    Notes:
    - Expects an autoregressive causal LM (`nn.Module`) with a forward method
      that accepts `input_ids` and returns logits (`FloatTensor [B, T, V]`).
    - A Hugging Face tokenizer-like object is required with attributes
      `eos_token_id`, `pad_token_id` and a callable interface returning
      `{"input_ids": LongTensor}` when invoked on a list of strings.
    - The reward function should accept: `(response: str, prompt: str, context: Dict[str, Any])`
      and return either a float or `{ "reward": float, "reward_info": Dict[str, float] }`.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        prompts: List[Union[str, Dict[str, Any]]],
        reward_fn: Callable[[str, str, Dict[str, Any]], Union[float, Dict[str, Any]]],
        *,
        # Sampling
        max_gen_len: int = 64,
        num_questions_per_batch: int = 4,
        num_answers_per_question: int = 2,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        # Optimization
        learning_rate: float = 1e-5,
        weight_decay: float = 0.0,
        adam_betas: tuple = (0.9, 0.999),
        adam_eps: float = 1e-8,
        max_grad_norm: float = 1.0,
        micro_batch_size: int = 8,
        total_steps: int = 100,
        # Mixed precision
        fp16: bool = False,
        bf16: bool = True,
        # Logging
        logging_backend: str = "tensorboard",
        logging_steps: int = 10,
        output_dir: str = "./outputs",
        run_name: Optional[str] = None,
        # Misc
        seed: int = 42,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.prompts = prompts
        self.reward_fn = reward_fn

        self.max_gen_len = max_gen_len
        self.num_questions_per_batch = max(1, num_questions_per_batch)
        self.num_answers_per_question = max(1, num_answers_per_question)
        self.temperature = max(1e-8, float(temperature))
        self.top_k = int(top_k)
        self.top_p = float(top_p)

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.adam_betas = adam_betas
        self.adam_eps = adam_eps
        self.max_grad_norm = max_grad_norm
        self.micro_batch_size = max(1, micro_batch_size)
        self.total_steps = total_steps

        self.fp16 = fp16
        self.bf16 = bf16

        self.logging_backend = logging_backend
        self.logging_steps = logging_steps
        self.output_dir = Path(output_dir)
        self.run_name = run_name

        self.device = get_device(prefer_cuda=True)
        self.model = self.model.to(self.device)
        self.model.train()

        set_seed(seed)

        self.loggers = setup_logging(
            backend=self.logging_backend,
            project_name="tinyft_grpo",
            run_name=self.run_name,
            log_dir=str(self.output_dir / "logs"),
        )

        self.global_step = 0
        self.memory_tracker = MemoryTracker(self.device)
        self.scaler = None
        if self.fp16:
            self.scaler = torch.amp.GradScaler("cuda")

        # Basic AdamW optimizer
        self.optimizer = AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.learning_rate,
            betas=self.adam_betas,
            eps=self.adam_eps,
            weight_decay=self.weight_decay,
        )

        # Cache common token ids
        self.pad_token_id = getattr(self.tokenizer, "pad_token_id", 0)
        self.eos_token_id = getattr(self.tokenizer, "eos_token_id", None)


    def train(self) -> Dict[str, Any]:
        """Run GRPO training over `total_steps` sampling updates."""
        metrics: Dict[str, Any] = {}
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Build simple dataloader over prompts
        # Each item may be a str or a dict with keys: 'prompt' and optional context
        prompt_items = self.prompts

        step_pbar = tqdm(range(self.total_steps), desc="GRPO Steps", position=0)
        for _ in step_pbar:
            batch_prompts, batch_context = self._sample_batch(prompt_items)

            # Rollout multiple answers per prompt
            episodes = self._rollout(batch_prompts, batch_context)

            # Update policy using token-wise policy gradient
            step_metrics = self._update_policy(episodes)

            self.global_step += 1
            metrics = {**step_metrics}

            # Logging
            if self.global_step % self.logging_steps == 0:
                self._log_metrics(metrics, self.global_step)

            # Progress bar
            cur_mem = self.memory_tracker.update()
            step_pbar.set_postfix({
                "loss": f"{metrics.get('loss', 0.0):.4f}",
                "entropy": f"{metrics.get('entropy', 0.0):.3f}",
                "mem": f"{cur_mem:.0f}MB",
            })

        # Save final model checkpoint
        self.save_checkpoint("grpo_final_model")
        return metrics

    def save_checkpoint(self, checkpoint_name: str) -> None:
        save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            epoch=0,
            step=self.global_step,
            loss=0.0,
            save_path=self.output_dir / f"{checkpoint_name}.pt",
            metadata={"trainer": "TinyGRPOTrainer"},
        )

    def _log_metrics(self, metrics: Dict[str, Any], step: int):
        """Log metrics to configured backends"""
        # TensorBoard logging
        if hasattr(self, 'tb_logger') and self.tb_logger is not None:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.tb_logger.add_scalar(key, value, step)
        
        # Basic console logging for missing metrics
        if step % self.logging_steps == 0:
            metrics_str = ", ".join([f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" 
                                   for k, v in metrics.items()])
            print(f"Step {step}: {metrics_str}")

    # ----------------------- Core algorithm ----------------------
    def _sample_batch(
        self, items: List[Union[str, Dict[str, Any]]]
    ) -> tuple[List[str], List[Dict[str, Any]]]:
        """Randomly sample a batch of prompts and their contexts."""
        import random

        batch_prompts: List[str] = []
        batch_context: List[Dict[str, Any]] = []
        for _ in range(self.num_questions_per_batch):
            entry = random.choice(items)
            if isinstance(entry, str):
                batch_prompts.append(entry)
                batch_context.append({})
            else:
                prompt = entry.get("prompt") or entry.get("question") or entry.get("text")
                if prompt is None:
                    raise ValueError("Prompt dict must include a 'prompt' key")
                ctx = {k: v for k, v in entry.items() if k not in {"prompt", "question", "text"}}
                batch_prompts.append(prompt)
                batch_context.append(ctx)
        return batch_prompts, batch_context

    @torch.no_grad()
    def _rollout(self, prompts: List[str], context: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate multiple responses per prompt and compute rewards.

        Returns a list of episode dicts with fields:
        - 'prompt': str
        - 'prompt_ids': List[int]
        - 'generated_ids': List[int]
        - 'is_finished': bool
        - 'reward': float
        - 'reward_info': Dict[str, float]
        - 'context': Dict[str, Any]
        """
        # Normalize prompts to plain text (handle chat/message formats)
        def _prompt_to_text(p: Any) -> str:
            # Already a string
            if isinstance(p, str):
                return p

            # Dictionary forms
            if isinstance(p, dict):
                # Common textual fields
                for key in ("prompt", "question", "text", "input"):
                    if key in p:
                        return _prompt_to_text(p[key])
                # Chat messages under 'messages'
                if "messages" in p:
                    msgs = p["messages"]
                    if hasattr(self.tokenizer, "apply_chat_template"):
                        return self.tokenizer.apply_chat_template(
                            msgs, tokenize=False, add_generation_prompt=True
                        )
                    # Fallback: join message contents
                    return "\n".join(
                        (m.get("content", "") if isinstance(m, dict) else str(m))
                        for m in msgs
                    )
                # Fallback
                return str(p)

            # List/Tuple forms (possible chat list[{role, content}] or list[str])
            if isinstance(p, (list, tuple)):
                if p and isinstance(p[0], dict) and "content" in p[0]:
                    if hasattr(self.tokenizer, "apply_chat_template"):
                        return self.tokenizer.apply_chat_template(
                            p, tokenize=False, add_generation_prompt=True
                        )
                    return "\n".join(
                        (m.get("content", "") if isinstance(m, dict) else str(m))  # type: ignore
                        for m in p
                    )
                if all(isinstance(x, str) for x in p):
                    return " ".join(p)  # type: ignore[arg-type]
                return str(p)

            # Anything else
            return str(p)

        prompt_texts: List[str] = [_prompt_to_text(p) for p in prompts]

        # Tokenize prompts (list[str] -> list[tensor])
        encoded = self._encode_prompts(prompt_texts)
        prefix_ids_list: List[List[int]] = [ids.tolist() for ids in encoded]

        # Build batch of size Q * M
        Q = len(prompts)
        M = self.num_answers_per_question
        bsz = Q * M
        min_prompt_len = min(len(t) for t in prefix_ids_list)
        max_prompt_len = max(len(t) for t in prefix_ids_list)
        total_len = max_prompt_len + self.max_gen_len

        tokens = torch.full(
            (bsz, total_len),
            fill_value=self.pad_token_id,
            dtype=torch.long,
            device=self.device,
        )
        for k, t in enumerate(prefix_ids_list):
            offset = k * M
            for i in range(M):
                tokens[offset + i, : len(t)] = torch.tensor(t, dtype=torch.long, device=self.device)

        input_text_mask = tokens != self.pad_token_id
        is_finished = torch.zeros((bsz,), dtype=torch.bool, device=self.device)

        prev_pos = 0
        for cur_pos in range(min_prompt_len, total_len):
            # Compute logits; naive full-prefix forward to keep dependencies minimal
            logits = self._model_logits(tokens[:, prev_pos:cur_pos])
            logits_last = logits[:, -1, :]
            next_token = self._sample_from_logits(logits_last)
            # Keep prompt tokens unchanged
            next_token = torch.where(input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token)
            # Respect EOS; fill pad after finished
            if self.eos_token_id is not None:
                is_end = next_token == int(self.eos_token_id)
                is_generated = ~input_text_mask[:, cur_pos]
                is_finished = is_finished | (is_end & is_generated)
                next_token = torch.where(is_finished, torch.tensor(self.pad_token_id, device=self.device), next_token)
            tokens[:, cur_pos] = next_token
            prev_pos = cur_pos
            if bool(is_finished.all().item()):
                break

        # Build episodes
        episodes: List[Dict[str, Any]] = []
        tokens_list = tokens.tolist()
        pad_id = int(self.pad_token_id)
        for i in range(Q):
            for j in range(M):
                idx = i * M + j
                prompt_ids = prefix_ids_list[i]
                gen_ids = tokens_list[idx][len(prompt_ids) :]
                if pad_id in gen_ids:
                    gen_ids = gen_ids[: gen_ids.index(pad_id)]
                response_text = self._decode_ids(gen_ids)
                # Reward
                r = self.reward_fn(response_text, prompt_texts[i], context[i])
                if isinstance(r, dict):
                    reward = float(r.get("reward", 0.0))
                    reward_info = {"reward": reward, **{k: float(v) for k, v in r.get("reward_info", {}).items()}}
                else:
                    reward = float(r)
                    reward_info = {"reward": reward}
                episodes.append(
                    {
                        "prompt": prompt_texts[i],
                        "prompt_ids": prompt_ids,
                        "generated_ids": gen_ids,
                        "is_finished": bool(is_finished[idx].item()),
                        "reward": reward,
                        "reward_info": reward_info,
                        "context": context[i],
                    }
                )
        # Normalize reward per group (by prompt)
        episodes = self._normalize_rewards_per_group(episodes)
        return episodes

    def _update_policy(self, episodes: List[Dict[str, Any]]) -> Dict[str, float]:
        """Compute token-wise policy gradient and update the policy once."""
        # Sort by total length for efficiency
        episodes_sorted = sorted(episodes, key=lambda e: len(e["prompt_ids"]) + len(e["generated_ids"]))
        num_target_tokens = sum(len(e["generated_ids"]) for e in episodes_sorted)
        if num_target_tokens == 0:
            return {"loss": 0.0, "grad_norm": 0.0, "entropy": 0.0}

        total_entropy = 0.0
        losses: List[torch.Tensor] = []

        for i in range(0, len(episodes_sorted), self.micro_batch_size):
            batch = episodes_sorted[i : i + self.micro_batch_size]
            batch_lengths = [len(e["prompt_ids"]) + len(e["generated_ids"]) for e in batch]
            max_len = max(batch_lengths)

            # Build token ids and masks
            token_ids = []
            masks = []
            advantages = []
            for k, e in enumerate(batch):
                seq = e["prompt_ids"] + e["generated_ids"]
                pad_n = max_len - len(seq)
                token_ids.append(seq + [self.pad_token_id] * pad_n)
                mask = [0] * len(e["prompt_ids"]) + [1] * len(e["generated_ids"]) + [0] * (pad_n)
                masks.append(mask)
                advantages.append(float(e["reward"]))

            token_ids_t = torch.tensor(token_ids, device=self.device, dtype=torch.long)
            masks_t = torch.tensor(masks, device=self.device, dtype=torch.bool)
            adv_t = torch.tensor(advantages, device=self.device, dtype=torch.float32)

            input_ids = token_ids_t[:, :-1]
            target_ids = token_ids_t[:, 1:]
            target_masks = masks_t[:, 1:]

            logits = self._model_logits(input_ids).float()

            # Compute per-token log-probabilities
            log_probs = -torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                target_ids.reshape(-1),
                ignore_index=self.pad_token_id,
                reduction="none",
            ).reshape(input_ids.shape[0], -1)

            # Entropy for logging
            with torch.no_grad():
                probs = torch.softmax(logits, dim=-1)
                token_entropy = torch.logsumexp(logits, dim=-1) - torch.sum(probs * logits, dim=-1)
                total_entropy += float((token_entropy * target_masks).sum().item())

            # Policy gradient objective per token
            obj = log_probs * adv_t[:, None]
            obj = (obj * target_masks).sum() / max(1, num_target_tokens)
            loss = -obj

            if self.fp16 and self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            losses.append(loss.detach())

        # Optimizer step (single step for the whole episodes set)
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        if self.fp16 and self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

        mean_loss = torch.stack([l if torch.is_tensor(l) else torch.tensor(l) for l in losses]).mean().item() if losses else 0.0
        return {
            "loss": float(mean_loss),
            "grad_norm": float(grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm),
            "entropy": float(total_entropy / max(1, num_target_tokens)),
        }

    # --------------------------- Helpers -------------------------
    def _encode_prompts(self, prompts: List[str]) -> List[torch.Tensor]:
        # Support HF tokenizers that accept batch encoding
        encoded = self.tokenizer(
            prompts,
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_tensors=None,
        )
        # HF returns dict-like object (BatchEncoding) with 'input_ids'; custom may return list[list[int]]
        input_ids = None
        if isinstance(encoded, dict) or hasattr(encoded, "get"):
            input_ids = encoded.get("input_ids")
            if isinstance(input_ids, torch.Tensor):
                return [row for row in input_ids]
            elif isinstance(input_ids, list):
                return [torch.tensor(row, dtype=torch.long, device=self.device) for row in input_ids]
        # Fallback: assume encoded is list[list[int]]
        assert isinstance(encoded, list)
        return [torch.tensor(row, dtype=torch.long, device=self.device) for row in encoded]

    def _decode_ids(self, ids: List[int]) -> str:
        # Try HF decode
        if hasattr(self.tokenizer, "decode"):
            try:
                return self.tokenizer.decode(ids, skip_special_tokens=True)
            except TypeError:
                return self.tokenizer.decode(ids)
        # Fallback: naive join
        return " ".join(str(i) for i in ids)

    def _model_logits(self, token_ids: torch.Tensor) -> torch.Tensor:
        # Expect model to return either logits directly or an object with .logits
        outputs = self.model(input_ids=token_ids)
        if isinstance(outputs, torch.Tensor):
            return outputs
        logits = getattr(outputs, "logits", None)
        if logits is None:
            raise RuntimeError("Model forward must return logits or object with .logits")
        return logits

    def _sample_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        # Temperature scaling
        scaled = logits / self.temperature

        # Top-k
        if self.top_k and self.top_k > 0:
            v, ix = torch.topk(scaled, self.top_k)
            mask = torch.full_like(scaled, float("-inf"))
            mask.scatter_(1, ix, v)
            scaled = mask

        # Top-p nucleus sampling
        if self.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(scaled, descending=True)
            probs = torch.softmax(sorted_logits, dim=-1)
            cumulative = torch.cumsum(probs, dim=-1)
            mask = cumulative > self.top_p
            mask[..., 1:] = mask[..., :-1].clone()
            mask[..., 0] = False
            sorted_logits[mask] = float("-inf")
            # Unsort
            unsorted = torch.full_like(scaled, float("-inf"))
            unsorted.scatter_(1, sorted_indices, sorted_logits)
            scaled = unsorted

        probs = torch.softmax(scaled, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
        return next_token

    @staticmethod
    def _normalize_rewards_per_group(episodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # Group by identical prompt string
        from collections import defaultdict

        grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for e in episodes:
            grouped[e["prompt"]].append(e)

        normed: List[Dict[str, Any]] = []
        for group in grouped.values():
            rewards = torch.tensor([float(e["reward"]) for e in group], dtype=torch.float32)
            mean = float(rewards.mean().item())
            std = float(rewards.std(unbiased=False).item())
            denom = std + 1e-4
            for e in group:
                e = {**e, "reward": (float(e["reward"]) - mean) / denom}
                normed.append(e)
        return normed
