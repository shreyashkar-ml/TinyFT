import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Any, Union, Callable
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class SFTDataset(Dataset):
    """
    Supervised Fine-Tuning Dataset
    
    Handles instruction-response datasets with proper formatting and tokenization
    for supervised fine-tuning tasks.
    """
    
    def __init__(
        self,
        data: List[Dict[str, Any]],
        tokenizer,
        max_length: int = 2048,
        instruction_column: str = "instruction",
        input_column: Optional[str] = "input",
        output_column: str = "output",
        response_template: str = "\n### Response:\n",
        ignore_index: int = -100
    ):
        """
        Initialize SFT dataset
        
        Args:
            data: List of data samples
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
            instruction_column: Column name for instructions
            input_column: Column name for input context (optional)
            output_column: Column name for expected outputs
            response_template: Template to separate instruction from response
            ignore_index: Index to ignore in loss computation
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.instruction_column = instruction_column
        self.input_column = input_column
        self.output_column = output_column
        self.response_template = response_template
        self.ignore_index = ignore_index
        
        # Ensure tokenizer has pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    @classmethod
    def from_hf(
        cls,
        dataset_name: str,
        tokenizer,
        split: str = "train",
        **kwargs
    ) -> "SFTDataset":
        """
        Create dataset from HuggingFace dataset
        
        Args:
            dataset_name: HuggingFace dataset name
            tokenizer: Tokenizer to use
            split: Dataset split to use
            **kwargs: Additional arguments for dataset creation
            
        Returns:
            SFTDataset instance
        """
        try:
            from datasets import load_dataset
            
            dataset = load_dataset(dataset_name, split=split)
            data = [dict(sample) for sample in dataset]
            
            logger.info(f"Loaded {len(data)} samples from {dataset_name}")
            return cls(data, tokenizer, **kwargs)
            
        except ImportError:
            raise ImportError("datasets library required. Install with: pip install datasets")
    
    @classmethod
    def from_json(
        cls,
        json_data: List[Dict[str, Any]],
        tokenizer,
        **kwargs
    ) -> "SFTDataset":
        """
        Create dataset from JSON data
        
        Args:
            json_data: List of dictionaries containing the data
            tokenizer: Tokenizer to use
            **kwargs: Additional arguments for dataset creation
            
        Returns:
            SFTDataset instance
        """
        logger.info(f"Creating SFT dataset from {len(json_data)} JSON samples")
        return cls(json_data, tokenizer, **kwargs)
    
    @classmethod
    def from_text_file(
        cls,
        file_path: str,
        tokenizer,
        **kwargs
    ) -> "SFTDataset":
        """
        Create dataset from text file (JSON Lines format)
        
        Args:
            file_path: Path to text file containing JSON lines
            tokenizer: Tokenizer to use
            **kwargs: Additional arguments for dataset creation
            
        Returns:
            SFTDataset instance
        """
        try:
            import json
            data = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data.append(json.loads(line))
            logger.info(f"Loaded {len(data)} samples from {file_path}")
            return cls(data, tokenizer, **kwargs)
        except FileNotFoundError:
            raise FileNotFoundError(f"Text file not found: {file_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in file {file_path}: {e}")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single training example"""
        sample = self.data[idx]
        
        # Build instruction text
        instruction = sample[self.instruction_column]
        if self.input_column and sample.get(self.input_column):
            instruction = f"{instruction}\n\nInput: {sample[self.input_column]}"
        
        # Build full text with response
        response = sample[self.output_column]
        full_text = f"{instruction}{self.response_template}{response}"
        
        # Tokenize full text
        tokenized = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Create labels (same as input_ids but with instruction part masked)
        labels = tokenized["input_ids"].clone()
        
        # Find where response starts and mask instruction part
        response_start = self._find_response_start(full_text, instruction)
        if response_start > 0:
            # Tokenize just the instruction part to find where to start labels
            instruction_tokens = self.tokenizer(
                instruction + self.response_template,
                truncation=True,
                max_length=self.max_length,
                add_special_tokens=False
            )
            instruction_length = len(instruction_tokens["input_ids"])
            
            # Mask instruction tokens in labels
            labels[0, :instruction_length] = self.ignore_index
        
        # Mask padding tokens in labels
        labels[tokenized["input_ids"] == self.tokenizer.pad_token_id] = self.ignore_index
        
        return {
            "input_ids": tokenized["input_ids"].squeeze(0),
            "attention_mask": tokenized["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0)
        }
    
    def _find_response_start(self, full_text: str, instruction: str) -> int:
        """Find where the response starts in the full text"""
        response_pos = full_text.find(self.response_template)
        return response_pos + len(self.response_template) if response_pos != -1 else 0


class CPTDataset(Dataset):
    """
    Continued Pre-Training Dataset
    
    Handles raw text data for continued pre-training with proper chunking
    and tokenization.
    """
    
    def __init__(
        self,
        data: List[str],
        tokenizer,
        max_length: int = 2048,
        stride: int = 1024,
        min_length: int = 512
    ):
        """
        Initialize CPT dataset
        
        Args:
            data: List of text strings
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
            stride: Stride for sliding window chunking
            min_length: Minimum chunk length to keep
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        self.min_length = min_length
        
        # Ensure tokenizer has pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Process and chunk the data
        self.chunks = self._process_data(data)
        logger.info(f"Created {len(self.chunks)} chunks from {len(data)} documents")
    
    @classmethod
    def from_hf(
        cls,
        dataset_name: str,
        tokenizer,
        text_column: str = "text",
        split: str = "train",
        **kwargs
    ) -> "CPTDataset":
        """
        Create dataset from HuggingFace dataset
        
        Args:
            dataset_name: HuggingFace dataset name
            tokenizer: Tokenizer to use
            text_column: Column containing text data
            split: Dataset split to use
            **kwargs: Additional arguments for dataset creation
            
        Returns:
            CPTDataset instance
        """
        try:
            from datasets import load_dataset
            
            dataset = load_dataset(dataset_name, split=split)
            data = [sample[text_column] for sample in dataset]
            
            logger.info(f"Loaded {len(data)} documents from {dataset_name}")
            return cls(data, tokenizer, **kwargs)
            
        except ImportError:
            raise ImportError("datasets library required. Install with: pip install datasets")
    
    @classmethod
    def from_text_file(
        cls,
        file_path: str,
        tokenizer,
        **kwargs
    ) -> "CPTDataset":
        """
        Create dataset from text file
        
        Args:
            file_path: Path to text file
            tokenizer: Tokenizer to use
            **kwargs: Additional arguments for dataset creation
            
        Returns:
            CPTDataset instance
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            data = [text]  # Treat whole file as one document
            logger.info(f"Loaded text file: {file_path}")
            return cls(data, tokenizer, **kwargs)
        except FileNotFoundError:
            raise FileNotFoundError(f"Text file not found: {file_path}")
    
    @classmethod
    def from_text_list(
        cls,
        text_list: List[str],
        tokenizer,
        **kwargs
    ) -> "CPTDataset":
        """
        Create dataset from list of text strings
        
        Args:
            text_list: List of text strings
            tokenizer: Tokenizer to use
            **kwargs: Additional arguments for dataset creation
            
        Returns:
            CPTDataset instance
        """
        logger.info(f"Creating CPT dataset from {len(text_list)} text samples")
        return cls(text_list, tokenizer, **kwargs)
    
    def _process_data(self, data: List[str]) -> List[str]:
        """Process raw text data into chunks"""
        chunks = []
        
        for text in data:
            # Tokenize the full text
            tokens = self.tokenizer(text, add_special_tokens=False)["input_ids"]
            
            # Create overlapping chunks
            for i in range(0, len(tokens), self.stride):
                chunk_tokens = tokens[i:i + self.max_length]
                
                # Skip chunks that are too short
                if len(chunk_tokens) < self.min_length:
                    continue
                
                # Decode back to text
                chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
                chunks.append(chunk_text)
        
        return chunks
    
    def __len__(self) -> int:
        return len(self.chunks)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single training example"""
        text = self.chunks[idx]
        
        # Tokenize
        tokenized = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        # For CPT, labels are the same as input_ids (standard language modeling)
        labels = tokenized["input_ids"].clone()
        
        # Mask padding tokens in labels
        labels[tokenized["input_ids"] == self.tokenizer.pad_token_id] = -100
        
        return {
            "input_ids": tokenized["input_ids"].squeeze(0),
            "attention_mask": tokenized["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0)
        }


class DatasetBuilder:
    """
    Factory class for creating datasets
    
    Provides convenient methods for creating different types of datasets
    with common configurations.
    """
    
    @staticmethod
    def create_sft_dataset(
        dataset_name: str,
        tokenizer,
        **kwargs
    ) -> SFTDataset:
        """Create an SFT dataset"""
        return SFTDataset.from_hf(dataset_name, tokenizer, **kwargs)
    
    @staticmethod
    def create_cpt_dataset(
        dataset_name: str,
        tokenizer,
        **kwargs
    ) -> CPTDataset:
        """Create a CPT dataset"""
        return CPTDataset.from_hf(dataset_name, tokenizer, **kwargs)
    
    @staticmethod
    def create_dataloader(
        dataset: Dataset,
        batch_size: int = 8,
        shuffle: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True,
        **kwargs
    ) -> DataLoader:
        """Create a DataLoader with common settings"""
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            **kwargs
        )


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for batching
    
    Args:
        batch: List of samples from dataset
        
    Returns:
        Batched tensors
    """
    # Stack all tensors
    batched = {}
    for key in batch[0].keys():
        batched[key] = torch.stack([sample[key] for sample in batch])
    
    return batched


def get_dataset_info(dataset: Dataset) -> Dict[str, Any]:
    """
    Get information about a dataset
    
    Args:
        dataset: Dataset to analyze
        
    Returns:
        Dictionary with dataset information
    """
    info = {
        "type": dataset.__class__.__name__,
        "size": len(dataset),
    }
    
    if hasattr(dataset, 'tokenizer'):
        info["tokenizer"] = dataset.tokenizer.__class__.__name__
        info["vocab_size"] = dataset.tokenizer.vocab_size
    
    if hasattr(dataset, 'max_length'):
        info["max_length"] = dataset.max_length
    
    # Get sample to analyze
    if len(dataset) > 0:
        sample = dataset[0]
        info["sample_keys"] = list(sample.keys())
        info["sample_shapes"] = {k: v.shape for k, v in sample.items()}
    
    return info 