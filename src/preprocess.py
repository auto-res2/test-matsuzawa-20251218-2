
"""
Complete preprocessing pipeline for datasets (CIFAR-10, ImageNet, GLUE/BERT).
Provides data loading, validation, and augmentation utilities.
"""

import logging
from pathlib import Path
from typing import Tuple, Optional, List

import torch
from torchvision import datasets, transforms

log = logging.getLogger(__name__)


# ============================================================================
# CIFAR-10 Preprocessing
# ============================================================================

class CIFAR10Preprocessor:
    """CIFAR-10 dataset preprocessing."""
    
    def __init__(self, cache_dir: str = ".cache", normalize: str = "standard_cifar10",
                augmentation: bool = True, pad: int = 4, crop_size: int = 32):
        """
        Initialize CIFAR-10 preprocessor.
        
        Args:
            cache_dir: Cache directory for downloaded data
            normalize: Normalization scheme ('standard_cifar10' or 'simple')
            augmentation: Whether to apply data augmentation
            pad: Padding size for crops
            crop_size: Final crop size
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.augmentation = augmentation
        self.pad = pad
        self.crop_size = crop_size
        
        # Normalization parameters
        if normalize == "standard_cifar10":
            self.mean = (0.4914, 0.4822, 0.4465)
            self.std = (0.2470, 0.2435, 0.2616)
        else:
            self.mean = (0.5, 0.5, 0.5)
            self.std = (0.5, 0.5, 0.5)
        
        log.info(f"CIFAR-10 Preprocessor initialized: augmentation={augmentation}, normalize={normalize}")
    
    def get_transforms(self, split: str = "train") -> transforms.Compose:
        """Get transforms for train/test splits."""
        
        normalize = transforms.Normalize(mean=self.mean, std=self.std)
        
        if split == "train" and self.augmentation:
            return transforms.Compose([
                transforms.Pad(self.pad),
                transforms.RandomCrop(self.crop_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            return transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
    
    def load_dataset(self, split: str = "train") -> datasets.CIFAR10:
        """Load CIFAR-10 dataset."""
        
        is_train = split == "train"
        transform = self.get_transforms(split)
        
        dataset = datasets.CIFAR10(
            root=str(self.cache_dir),
            train=is_train,
            download=True,
            transform=transform
        )
        
        log.info(f"Loaded CIFAR-10 ({split}): {len(dataset)} samples")
        return dataset


# ============================================================================
# ImageNet Preprocessing
# ============================================================================

class ImageNetPreprocessor:
    """ImageNet dataset preprocessing (ResNet-50)."""
    
    def __init__(self, cache_dir: str = ".cache", augmentation: bool = True):
        """
        Initialize ImageNet preprocessor.
        
        Args:
            cache_dir: Cache directory for dataset
            augmentation: Whether to apply data augmentation
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.augmentation = augmentation
        
        # Standard ImageNet normalization
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        
        log.info(f"ImageNet Preprocessor initialized: augmentation={augmentation}")
    
    def get_transforms(self, split: str = "train") -> transforms.Compose:
        """Get transforms for train/val splits (ResNet standard)."""
        
        normalize = transforms.Normalize(mean=self.mean, std=self.std)
        
        if split == "train" and self.augmentation:
            return transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            return transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])
    
    def load_dataset(self, split: str = "train", data_root: str = "/data/imagenet") -> Optional[torch.utils.data.Dataset]:
        """
        Load ImageNet dataset (requires data to be already downloaded).
        
        Args:
            split: 'train' or 'val'
            data_root: Root directory containing ImageNet data
        
        Returns:
            Dataset or None if path doesn't exist
        """
        
        data_path = Path(data_root) / ("train" if split == "train" else "val")
        
        if not data_path.exists():
            log.warning(f"ImageNet path not found: {data_path}")
            log.warning("Expected structure: /data/imagenet/train/ and /data/imagenet/val/")
            return None
        
        transform = self.get_transforms(split)
        
        dataset = datasets.ImageFolder(str(data_path), transform=transform)
        log.info(f"Loaded ImageNet ({split}): {len(dataset)} samples")
        
        return dataset


# ============================================================================
# GLUE/BERT Preprocessing
# ============================================================================

class GLUEPreprocessor:
    """GLUE dataset preprocessing for BERT fine-tuning."""
    
    def __init__(self, cache_dir: str = ".cache", task: str = "squad",
                max_seq_length: int = 384, doc_stride: int = 128):
        """
        Initialize GLUE preprocessor.
        
        Args:
            cache_dir: Cache directory
            task: Task name (squad, rte, etc.)
            max_seq_length: Maximum sequence length for tokenization
            doc_stride: Document stride for sliding window (QA tasks)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.task = task.lower()
        self.max_seq_length = max_seq_length
        self.doc_stride = doc_stride
        
        try:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                "bert-base-uncased",
                cache_dir=str(self.cache_dir)
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            log.info(f"Tokenizer initialized with pad_token_id={self.tokenizer.pad_token_id}")
        except ImportError:
            log.warning("transformers not available, tokenizer will be None")
            self.tokenizer = None
        
        log.info(f"GLUE Preprocessor initialized: task={task}, max_seq_length={max_seq_length}")
    
    def load_squad_dataset(self, split: str = "train") -> Optional[torch.utils.data.Dataset]:
        """Load SQuAD 2.0 dataset."""
        
        try:
            from datasets import load_dataset
        except ImportError:
            log.error("datasets library required for GLUE preprocessing")
            return None
        
        try:
            dataset = load_dataset(
                "squad_v2" if "2" in str(self.task) else "squad",
                cache_dir=str(self.cache_dir)
            )
            
            split_name = "validation" if split == "val" else split
            if split_name not in dataset:
                split_name = list(dataset.keys())[0]
            
            split_data = dataset[split_name]
            log.info(f"Loaded SQuAD ({split}): {len(split_data)} examples")
            
            return split_data
        
        except Exception as e:
            log.error(f"Failed to load SQuAD: {e}")
            return None
    
    def load_rte_dataset(self, split: str = "train") -> Optional[torch.utils.data.Dataset]:
        """Load RTE dataset for GLUE."""
        
        try:
            from datasets import load_dataset
        except ImportError:
            log.error("datasets library required for GLUE preprocessing")
            return None
        
        try:
            dataset = load_dataset(
                "rte",
                cache_dir=str(self.cache_dir)
            )
            
            split_name = "validation" if split == "val" else split
            if split_name not in dataset:
                split_name = list(dataset.keys())[0]
            
            split_data = dataset[split_name]
            log.info(f"Loaded RTE ({split}): {len(split_data)} examples")
            
            return split_data
        
        except Exception as e:
            log.error(f"Failed to load RTE: {e}")
            return None


# ============================================================================
# Factory Function
# ============================================================================

def get_preprocessor(dataset_name: str, cache_dir: str = ".cache", **kwargs):
    """
    Factory function to get appropriate preprocessor.
    
    Args:
        dataset_name: Name of dataset (CIFAR-10, ImageNet, SQuAD, RTE, etc.)
        cache_dir: Cache directory
        **kwargs: Additional arguments for preprocessor
    
    Returns:
        Preprocessor instance
    """
    
    dataset_name_lower = dataset_name.lower()
    
    if dataset_name_lower in ["cifar-10", "cifar10"]:
        return CIFAR10Preprocessor(cache_dir=cache_dir, **kwargs)
    
    elif dataset_name_lower in ["imagenet"]:
        return ImageNetPreprocessor(cache_dir=cache_dir, **kwargs)
    
    elif dataset_name_lower in ["squad", "squad2", "rte", "glue"]:
        return GLUEPreprocessor(cache_dir=cache_dir, task=dataset_name_lower, **kwargs)
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    log.info("Testing preprocessing pipelines...")
    
    # Test CIFAR-10
    try:
        cifar_prep = get_preprocessor("CIFAR-10", cache_dir=".cache")
        train_data = cifar_prep.load_dataset("train")
        test_data = cifar_prep.load_dataset("test")
        log.info(f"✓ CIFAR-10: {len(train_data)} train, {len(test_data)} test")
    except Exception as e:
        log.error(f"CIFAR-10 test failed: {e}")
    
    # Test ImageNet (will skip if data not available)
    try:
        imagenet_prep = get_preprocessor("ImageNet", cache_dir=".cache")
        log.info("✓ ImageNet preprocessor created (data availability depends on system)")
    except Exception as e:
        log.error(f"ImageNet test failed: {e}")
    
    # Test GLUE
    try:
        glue_prep = get_preprocessor("SQuAD", cache_dir=".cache")
        log.info("✓ GLUE preprocessor created")
    except Exception as e:
        log.error(f"GLUE test failed: {e}")
