
"""
Single experiment run executor with complete training pipeline.
Loads run configuration, trains model, and logs metrics to WandB.

Executed as subprocess from main.py with Hydra configuration.
"""

import os
import sys
import logging
import json
import time
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

# Conditional imports
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

try:
    import optuna
    from optuna.samplers import TPESampler
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

log = logging.getLogger(__name__)


# ============================================================================
# SNARL Optimizer Implementation
# ============================================================================

class SNARL(torch.optim.Optimizer):
    """
    Signal-to-Noise Adaptive Learning Rate (SNARL) optimizer.
    
    Replaces RAdam's global variance rectification with per-layer SNR-based
    adaptive gating for faster early-stage convergence.
    """
    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, snr_beta=1.0, snr_warmup_steps=100):
        """
        Initialize SNARL optimizer.
        
        Args:
            params: Model parameters
            lr: Learning rate
            betas: Exponential moving average coefficients (beta1, beta2)
            eps: Numerical stability term
            weight_decay: L2 weight decay coefficient
            snr_beta: SNR sigmoid scaling factor
            snr_warmup_steps: Warmup iterations before SNR gating activation
        """
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        
        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
            snr_beta=snr_beta, snr_warmup_steps=snr_warmup_steps
        )
        super(SNARL, self).__init__(params, defaults)
    
    def step(self, closure=None):
        """Perform a single optimization step."""
        
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('SNARL does not support sparse gradients')
                
                p_data_fp32 = p.data.float()
                
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                    state['snr_history'] = []
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                
                # Apply weight decay
                if group['weight_decay'] != 0:
                    grad = grad.add(p_data_fp32, alpha=group['weight_decay'])
                
                # Update biased first and second moment estimates
                exp_avg_sq.mul_(beta2).add_(grad ** 2, alpha=1 - beta2)
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                
                state['step'] += 1
                
                # Bias correction factors
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                # ===== CORE INNOVATION: Per-layer SNR-based adaptive gating =====
                
                # Compute SNR: signal (momentum) / noise (residual variance)
                momentum_norm_sq = torch.sum(exp_avg ** 2)
                variance = torch.sum(exp_avg_sq)
                noise_var = torch.clamp(variance - momentum_norm_sq, min=group['eps'])
                
                snr = momentum_norm_sq / (noise_var + group['eps'])
                snr_item = snr.item()
                
                # SNR-based adaptive gate using sigmoid
                snr_log = math.log(snr_item + 1.0)
                gate_logit = group['snr_beta'] * snr_log
                adaptive_gate = 1.0 / (1.0 + math.exp(-gate_logit))  # sigmoid
                
                # Warmup phase: gradually activate SNR gating
                if state['step'] < group['snr_warmup_steps']:
                    warmup_progress = state['step'] / group['snr_warmup_steps']
                    adaptive_gate = adaptive_gate * warmup_progress
                
                state['snr_history'].append(snr_item)
                
                # ===== Hybrid update: gated combination =====
                
                # Adaptive component (RAdam-style)
                adaptive_lr = math.sqrt(bias_correction2) / bias_correction1
                adaptive_update = adaptive_lr * exp_avg / (exp_avg_sq.sqrt() + group['eps'])
                
                # Momentum-only component
                momentum_update = exp_avg / bias_correction1
                
                # Gated combination
                final_update = (adaptive_gate * adaptive_update + 
                               (1 - adaptive_gate) * momentum_update)
                
                # Apply parameter update
                p_data_fp32.add_(-group['lr'] * final_update, alpha=1)
                p.data.copy_(p_data_fp32)
        
        return loss


# ============================================================================
# RAdam Optimizer (fallback implementation)
# ============================================================================

class RAdam(torch.optim.Optimizer):
    """
    Rectified Adam optimizer (fallback implementation if torch.optim doesn't have it).
    """
    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon: {eps}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid betas")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(RAdam, self).__init__(params, defaults)
    
    def step(self, closure=None):
        """Perform optimization step."""
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')
                
                p_data_fp32 = p.data.float()
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                
                # Apply weight decay
                if group['weight_decay'] != 0:
                    grad = grad.add(p_data_fp32, alpha=group['weight_decay'])
                
                # Update moments
                exp_avg_sq.mul_(beta2).add_(grad ** 2, alpha=1 - beta2)
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                
                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                # RAdam variance rectification
                beta2_t = beta2 ** state['step']
                N_sma_max = 2 / (1 - beta2) - 1
                N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                
                if N_sma >= 5:
                    rect = math.sqrt(
                        ((N_sma - 4) * (N_sma - 2) * N_sma_max) / 
                        ((N_sma_max - 4) * (N_sma_max - 2) * N_sma)
                    )
                else:
                    rect = 0.0
                
                if rect > 0:
                    adaptive_lr = math.sqrt(bias_correction2) / bias_correction1 * rect
                    update = adaptive_lr * exp_avg / (exp_avg_sq.sqrt() + group['eps'])
                else:
                    update = exp_avg / bias_correction1
                
                p_data_fp32.add_(-group['lr'] * update, alpha=1)
                p.data.copy_(p_data_fp32)
        
        return loss


# ============================================================================
# Utility Functions
# ============================================================================

def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Get device (GPU if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_convergence_speed(metrics_history: List[float], 
                             target_percentage: float = 0.95) -> int:
    """Compute convergence speed: epoch where optimizer reaches 95% of final performance."""
    if not metrics_history or len(metrics_history) == 0:
        return -1
    
    final_value = metrics_history[-1]
    if final_value <= 0:
        return -1
    
    target_threshold = target_percentage * final_value
    
    for epoch, metric in enumerate(metrics_history):
        if metric >= target_threshold:
            return epoch
    
    return len(metrics_history)


def extract_snr_statistics(optimizer: torch.optim.Optimizer) -> Dict[str, List[float]]:
    """Extract SNR history from SNARL optimizer state."""
    snr_stats = {}
    param_idx = 0
    
    for group in optimizer.param_groups:
        for p in group['params']:
            state = optimizer.state.get(p, {})
            if 'snr_history' in state:
                snr_stats[f"param_{param_idx}"] = state['snr_history']
                param_idx += 1
    
    return snr_stats


# ============================================================================
# Data Loading (CIFAR-10)
# ============================================================================

def load_cifar10_data(cfg: DictConfig, cache_dir: str = ".cache") -> Tuple:
    """Load CIFAR-10 dataset with preprocessing and train/val split."""
    
    from torchvision import datasets, transforms
    
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    
    # Normalization
    if cfg.dataset.preprocessing.normalization == "standard_cifar10":
        normalize = transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616]
        )
    else:
        normalize = transforms.Normalize((0.5,), (0.5,))
    
    # Training transforms
    if cfg.dataset.preprocessing.augmentation:
        train_transform = transforms.Compose([
            transforms.Pad(cfg.dataset.preprocessing.pad),
            transforms.RandomCrop(cfg.dataset.preprocessing.crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    
    # Test transforms (no augmentation)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    
    # Load full training dataset
    full_train = datasets.CIFAR10(
        root=str(cache_path),
        train=True,
        download=True,
        transform=train_transform
    )
    
    # Load test dataset
    test_dataset = datasets.CIFAR10(
        root=str(cache_path),
        train=False,
        download=True,
        transform=test_transform
    )
    
    # Split training into train/val
    train_size = int(cfg.dataset.split.train * len(full_train))
    val_size = len(full_train) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_train,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(cfg.training.seed)
    )
    
    log.info(f"Loaded CIFAR-10: {train_size} train, {val_size} val, {len(test_dataset)} test")
    
    return train_dataset, val_dataset, test_dataset


def get_dataloaders(cfg: DictConfig, cache_dir: str = ".cache") -> Tuple:
    """Create data loaders for configured dataset."""
    
    if cfg.dataset.name.lower() not in ["cifar-10", "cifar10"]:
        log.error(f"Dataset '{cfg.dataset.name}' not fully implemented (CIFAR-10 supported)")
        sys.exit(1)
    
    train_data, val_data, test_data = load_cifar10_data(cfg, cache_dir)
    
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


# ============================================================================
# Model Creation (ResNet-20 for CIFAR-10)
# ============================================================================

class BasicBlock(nn.Module):
    """Basic residual block for ResNet."""
    expansion = 1
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1,
                 downsample: Optional[nn.Module] = None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = torch.nn.functional.relu(out, inplace=True)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = torch.nn.functional.relu(out, inplace=True)
        return out


class ResNetCIFAR(nn.Module):
    """ResNet optimized for CIFAR-10/100."""
    
    def __init__(self, block, num_blocks: List[int], num_classes: int = 10):
        super(ResNetCIFAR, self).__init__()
        self.in_channels = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * block.expansion, num_classes)
        self._init_weights()
    
    def _make_layer(self, block, out_channels: int, num_blocks: int,
                   stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )
        
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # Post-init assertion
        assert self.fc.out_features == 10 or self.fc.out_features == 100, \
            f"FC layer output dimension should match num_classes, got {self.fc.out_features}"
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.nn.functional.relu(x, inplace=True)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def resnet20_cifar(num_classes: int = 10) -> ResNetCIFAR:
    """ResNet-20 for CIFAR-10/100."""
    return ResNetCIFAR(BasicBlock, [3, 3, 3], num_classes=num_classes)


def create_model_from_config(cfg: DictConfig) -> nn.Module:
    """Create model from configuration."""
    model_name = cfg.model.name.lower()
    num_classes = cfg.model.num_classes
    
    if model_name in ["resnet20", "resnet-20"]:
        model = resnet20_cifar(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model


# ============================================================================
# Optimizer Creation
# ============================================================================

def create_optimizer(cfg: DictConfig, model: nn.Module) -> torch.optim.Optimizer:
    """Create optimizer from configuration."""
    
    optimizer_name = cfg.training.optimizer.lower()
    lr = cfg.training.learning_rate
    betas = cfg.training.additional_params.get('betas', [0.9, 0.999])
    eps = cfg.training.additional_params.get('eps', 1e-8)
    weight_decay = cfg.training.weight_decay
    
    if optimizer_name == "snarl":
        snr_beta = cfg.training.additional_params.get('snr_beta', 1.0)
        snr_warmup_steps = cfg.training.additional_params.get('snr_warmup_steps', 100)
        optimizer = SNARL(
            model.parameters(),
            lr=lr,
            betas=tuple(betas),
            eps=eps,
            weight_decay=weight_decay,
            snr_beta=snr_beta,
            snr_warmup_steps=snr_warmup_steps
        )
        log.info(f"Created SNARL optimizer: snr_beta={snr_beta}, snr_warmup_steps={snr_warmup_steps}")
    
    elif optimizer_name == "radam":
        optimizer = RAdam(
            model.parameters(),
            lr=lr,
            betas=tuple(betas),
            eps=eps,
            weight_decay=weight_decay
        )
        log.info("Created RAdam optimizer")
    
    elif optimizer_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, betas=tuple(betas),
                              eps=eps, weight_decay=weight_decay)
        log.info("Created Adam optimizer")
    
    elif optimizer_name == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=lr, betas=tuple(betas),
                               eps=eps, weight_decay=weight_decay)
        log.info("Created AdamW optimizer")
    
    elif optimizer_name == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=betas[0],
                             weight_decay=weight_decay)
        log.info("Created SGD optimizer")
    
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    return optimizer


def create_scheduler(cfg: DictConfig, optimizer: torch.optim.Optimizer,
                    num_epochs: int) -> Optional[object]:
    """Create learning rate scheduler."""
    
    scheduler_name = cfg.training.scheduler.lower()
    
    if scheduler_name == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
        log.info(f"Created CosineAnnealingLR scheduler (T_max={num_epochs})")
    
    elif scheduler_name == "step":
        scheduler = StepLR(optimizer, step_size=num_epochs // 3, gamma=0.1)
        log.info(f"Created StepLR scheduler")
    
    elif scheduler_name == "constant":
        scheduler = None
        log.info("Using constant learning rate")
    
    else:
        log.warning(f"Scheduler '{scheduler_name}' not recognized, using constant LR")
        scheduler = None
    
    return scheduler


# ============================================================================
# Training Loop
# ============================================================================

def train_epoch(model: nn.Module, train_loader, optimizer: torch.optim.Optimizer,
               criterion: nn.Module, device: torch.device, epoch: int,
               trial_mode_batch_limit: Optional[int] = None) -> Tuple[float, float]:
    """Train for one epoch."""
    
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    batch_count = 0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        
        # Trial mode: limit batches
        if trial_mode_batch_limit is not None and batch_idx >= trial_mode_batch_limit:
            break
        
        inputs, targets = inputs.to(device), targets.to(device)
        
        # CRITICAL ASSERTION: Verify input/label shapes at start of batch
        if batch_idx == 0 and epoch == 0:
            assert inputs.shape[0] == targets.shape[0], \
                f"Batch size mismatch: inputs {inputs.shape[0]} vs targets {targets.shape[0]}"
            assert len(inputs.shape) == 4, \
                f"Expected 4D input tensor [N,C,H,W], got shape {inputs.shape}"
            assert len(targets.shape) == 1, \
                f"Expected 1D target tensor [N], got shape {targets.shape}"
        
        # Forward pass - NO LABEL CONCATENATION (data leak prevention)
        optimizer.zero_grad()
        outputs = model(inputs)
        
        # Post-forward assertions
        assert outputs.shape[0] == targets.shape[0], \
            f"Output batch size mismatch: {outputs.shape[0]} vs {targets.shape[0]}"
        assert not torch.isnan(outputs).any(), "NaN in model outputs"
        assert not torch.isinf(outputs).any(), "Inf in model outputs"
        
        # Loss computation ONLY (labels used here, not in model input)
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        
        # CRITICAL ASSERTION: Verify gradients exist and are non-zero before step
        grads_exist = False
        grads_nonzero = False
        
        for param in model.parameters():
            if param.grad is not None:
                grads_exist = True
                assert not torch.isnan(param.grad).any(), \
                    f"NaN detected in gradients at batch {batch_idx}"
                assert not torch.isinf(param.grad).any(), \
                    f"Inf detected in gradients at batch {batch_idx}"
                if (param.grad != 0).any():
                    grads_nonzero = True
        
        assert grads_exist and grads_nonzero, \
            f"Invalid gradients at batch {batch_idx}: " \
            f"grads_exist={grads_exist}, grads_nonzero={grads_nonzero}"
        
        # Optimizer step
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
        batch_count += 1
    
    avg_loss = total_loss / max(1, batch_count)
    avg_accuracy = 100 * correct / total if total > 0 else 0.0
    
    return avg_loss, avg_accuracy


def validate(model: nn.Module, val_loader, criterion: nn.Module,
            device: torch.device, trial_mode_batch_limit: Optional[int] = None) -> Tuple[float, float]:
    """Validate model on validation/test set."""
    
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    batch_count = 0
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            
            if trial_mode_batch_limit is not None and batch_idx >= trial_mode_batch_limit:
                break
            
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            batch_count += 1
    
    avg_loss = total_loss / max(1, batch_count)
    avg_accuracy = 100 * correct / total if total > 0 else 0.0
    
    return avg_loss, avg_accuracy


# ============================================================================
# Optuna Hyperparameter Optimization
# ============================================================================

def objective_function(trial, cfg: DictConfig, train_loader, val_loader,
                      device: torch.device) -> float:
    """Optuna objective: train and evaluate on validation set."""
    
    try:
        # Sample hyperparameters
        trial_cfg = OmegaConf.create(OmegaConf.to_container(cfg))
        
        for search_space in cfg.optuna.search_spaces:
            param_name = search_space.param_name
            dist_type = search_space.distribution_type
            
            try:
                if dist_type == "loguniform":
                    value = trial.suggest_float(
                        param_name,
                        search_space.low,
                        search_space.high,
                        log=True
                    )
                elif dist_type == "uniform":
                    value = trial.suggest_float(
                        param_name,
                        search_space.low,
                        search_space.high
                    )
                elif dist_type == "int":
                    value = trial.suggest_int(
                        param_name,
                        int(search_space.low),
                        int(search_space.high)
                    )
                else:
                    continue
                
                # Update config
                if param_name in ["snr_beta", "snr_warmup_steps"]:
                    trial_cfg.training.additional_params[param_name] = value
                elif param_name == "learning_rate":
                    trial_cfg.training.learning_rate = value
                elif param_name == "weight_decay":
                    trial_cfg.training.weight_decay = value
            
            except Exception:
                continue
        
        # Create fresh model
        model = create_model_from_config(trial_cfg)
        model = model.to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = create_optimizer(trial_cfg, model)
        
        # Quick training (2 batches only)
        model.train()
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            if batch_idx >= 2:
                break
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        
        # Validation (2 batches only)
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(val_loader):
                if batch_idx >= 2:
                    break
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        val_accuracy = 100 * correct / total if total > 0 else 0.0
        return val_accuracy
    
    except Exception as e:
        log.warning(f"Trial failed: {e}")
        return 0.0


def run_optuna_optimization(cfg: DictConfig, train_loader, val_loader,
                           device: torch.device) -> Optional[object]:
    """Run Optuna hyperparameter optimization."""
    
    if not cfg.optuna.enabled or cfg.optuna.n_trials <= 0:
        log.info("Optuna optimization disabled")
        return None
    
    if not HAS_OPTUNA:
        log.warning("Optuna not available, skipping hyperparameter optimization")
        return None
    
    log.info(f"Starting Optuna optimization with {cfg.optuna.n_trials} trials")
    
    sampler = TPESampler(seed=cfg.training.seed)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    try:
        study.optimize(
            lambda trial: objective_function(trial, cfg, train_loader, val_loader, device),
            n_trials=cfg.optuna.n_trials,
            show_progress_bar=False
        )
        
        best_trial = study.best_trial
        log.info(f"Best trial #{best_trial.number}: value={best_trial.value:.4f}")
        
        # Update config with best parameters
        for param_name, value in best_trial.params.items():
            if param_name in ["snr_beta", "snr_warmup_steps"]:
                cfg.training.additional_params[param_name] = value
            elif param_name == "learning_rate":
                cfg.training.learning_rate = value
            elif param_name == "weight_decay":
                cfg.training.weight_decay = value
        
        return study
    
    except Exception as e:
        log.error(f"Optuna failed: {e}")
        return None


# ============================================================================
# Main Training Pipeline
# ============================================================================

def train_model(cfg: DictConfig, model: nn.Module, train_loader, val_loader,
               test_loader, device: torch.device) -> Dict:
    """Main training loop with comprehensive metrics tracking."""
    
    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer(cfg, model)
    scheduler = create_scheduler(cfg, optimizer, cfg.training.epochs)
    
    model = model.to(device)
    
    metrics = {
        'train_losses': [],
        'train_accuracies': [],
        'val_losses': [],
        'val_accuracies': [],
        'test_loss': None,
        'test_accuracy': None,
        'convergence_speed': None,
        'snr_statistics': {},
        'training_time': 0.0
    }
    
    # Get trial mode batch limit
    trial_mode_batch_limit = None
    if os.getenv('SNARL_BATCH_LIMIT'):
        limit = int(os.getenv('SNARL_BATCH_LIMIT'))
        if limit > 0:
            trial_mode_batch_limit = limit
    
    # Post-init assertions
    assert model is not None, "Model is None"
    assert optimizer is not None, "Optimizer is None"
    assert hasattr(model, 'fc') or hasattr(model, 'classifier'), "Model missing classifier"
    log.info(f"✓ Post-init: Model={type(model).__name__}, Optimizer={type(optimizer).__name__}")
    
    log.info(f"Training config: epochs={cfg.training.epochs}, batch_size={cfg.training.batch_size}")
    
    start_time = time.time()
    
    for epoch in range(cfg.training.epochs):
        epoch_start = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch,
            trial_mode_batch_limit=trial_mode_batch_limit
        )
        metrics['train_losses'].append(train_loss)
        metrics['train_accuracies'].append(train_acc)
        
        # Validate
        val_loss, val_acc = validate(
            model, val_loader, criterion, device,
            trial_mode_batch_limit=trial_mode_batch_limit
        )
        metrics['val_losses'].append(val_loss)
        metrics['val_accuracies'].append(val_acc)
        
        # Step scheduler
        if scheduler is not None:
            scheduler.step()
        
        epoch_time = time.time() - epoch_start
        
        # Log per-epoch metrics to WandB
        if HAS_WANDB and cfg.wandb.mode != "disabled":
            try:
                log_dict = {
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'train_accuracy': train_acc,
                    'val_loss': val_loss,
                    'val_accuracy': val_acc,
                    'learning_rate': optimizer.param_groups[0]['lr'],
                    'epoch_time': epoch_time
                }
                
                if isinstance(optimizer, SNARL):
                    snr_stats = extract_snr_statistics(optimizer)
                    if snr_stats:
                        avg_snr = np.mean([np.mean(v) for v in snr_stats.values()])
                        log_dict['avg_snr'] = avg_snr
                
                wandb.log(log_dict, step=epoch)
            except Exception as e:
                log.warning(f"WandB logging failed: {e}")
        
        # Log to stdout
        if (epoch + 1) % max(1, cfg.training.epochs // 10) == 0 or epoch == 0:
            log.info(f"Epoch {epoch+1}/{cfg.training.epochs}: "
                    f"train_loss={train_loss:.4f}, train_acc={train_acc:.2f}% | "
                    f"val_loss={val_loss:.4f}, val_acc={val_acc:.2f}%")
    
    metrics['training_time'] = time.time() - start_time
    
    # Test evaluation
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    metrics['test_loss'] = test_loss
    metrics['test_accuracy'] = test_acc
    log.info(f"Test: loss={test_loss:.4f}, accuracy={test_acc:.2f}%")
    
    # Compute convergence speed
    metrics['convergence_speed'] = compute_convergence_speed(metrics['val_accuracies'])
    log.info(f"Convergence speed: epoch {metrics['convergence_speed']}")
    
    # Extract SNR statistics
    if isinstance(optimizer, SNARL):
        metrics['snr_statistics'] = extract_snr_statistics(optimizer)
    
    return metrics


# ============================================================================
# Main Entry Point
# ============================================================================

@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training entry point for single run."""
    
    log.info("=" * 80)
    log.info("SNARL Training Pipeline")
    log.info("=" * 80)

    if cfg.run is None:
        log.error("ERROR: run must be set")
        sys.exit(1)

    # Load run-specific configuration
    run_config_path = Path("config") / "runs" / f"{cfg.run}.yaml"
    if not run_config_path.exists():
        log.error(f"Run config not found: {run_config_path}")
        sys.exit(1)
    
    log.info(f"Loading run configuration: {run_config_path}")
    run_cfg = OmegaConf.load(run_config_path)
    
    # Merge configurations
    cfg = OmegaConf.merge(cfg, run_cfg)
    
    # Apply mode-specific overrides
    mode = os.getenv('SNARL_MODE', cfg.get('mode', 'full'))
    if mode == "trial":
        cfg.training.epochs = 1
        cfg.wandb.mode = "disabled"
        cfg.optuna.n_trials = 0
        log.info("Trial mode activated")
    elif mode == "full":
        cfg.wandb.mode = "online"

    log.info(f"Config: run_id={cfg.run}, method={cfg.method}, mode={mode}")
    
    # Set seed
    set_seed(cfg.training.seed)
    
    # Get device
    device = get_device()
    log.info(f"Device: {device}")
    
    # Load data
    cache_dir = cfg.dataset.get('cache_dir', '.cache')
    train_loader, val_loader, test_loader = get_dataloaders(cfg, cache_dir)
    log.info(f"Data loaded: {len(train_loader)} train, {len(val_loader)} val")
    
    # Initialize WandB
    wandb_enabled = HAS_WANDB and cfg.wandb.mode != "disabled"
    if wandb_enabled:
        try:
            wandb.init(
                entity=cfg.wandb.entity,
                project=cfg.wandb.project,
                id=cfg.run,
                config=OmegaConf.to_container(cfg, resolve=True),
                resume="allow",
                mode=cfg.wandb.mode
            )
            log.info(f"✓ WandB initialized: {wandb.run.url}")
            print(f"WandB Run URL: {wandb.run.url}")
        except Exception as e:
            log.warning(f"WandB init failed: {e}")
            wandb_enabled = False
    else:
        log.info("WandB disabled")
    
    # Optuna optimization (only if enabled and n_trials > 0)
    if cfg.optuna.enabled and cfg.optuna.n_trials > 0:
        log.info("Running Optuna optimization...")
        run_optuna_optimization(cfg, train_loader, val_loader, device)
        log.info("✓ Optuna complete")
    
    # Create model
    log.info(f"Creating model: {cfg.model.name}")
    model = create_model_from_config(cfg)
    total_params = sum(p.numel() for p in model.parameters())
    log.info(f"✓ Model created: {total_params:,} parameters")
    
    # Train
    log.info("Starting training...")
    metrics = train_model(cfg, model, train_loader, val_loader, test_loader, device)
    
    # Log summary to WandB
    if wandb_enabled:
        try:
            wandb.summary['final_test_accuracy'] = metrics['test_accuracy']
            wandb.summary['test_loss'] = metrics['test_loss']
            wandb.summary['convergence_speed'] = metrics['convergence_speed']
            wandb.summary['training_time'] = metrics['training_time']
            wandb.finish()
            log.info("✓ WandB finished")
        except Exception as e:
            log.warning(f"WandB finish failed: {e}")
    
    # Save metrics locally
    results_dir = Path(cfg.results_dir) / cfg.run
    results_dir.mkdir(parents=True, exist_ok=True)
    
    metrics_file = results_dir / "metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump({
            'train_losses': metrics['train_losses'],
            'train_accuracies': metrics['train_accuracies'],
            'val_losses': metrics['val_losses'],
            'val_accuracies': metrics['val_accuracies'],
            'test_loss': metrics['test_loss'],
            'test_accuracy': metrics['test_accuracy'],
            'convergence_speed': metrics['convergence_speed'],
            'training_time': metrics['training_time']
        }, f, indent=2)
    
    log.info(f"✓ Metrics saved: {metrics_file}")
    log.info("=" * 80)
    log.info("TRAINING COMPLETE")
    log.info(f"Test Accuracy: {metrics['test_accuracy']:.2f}%")
    log.info(f"Convergence Speed: epoch {metrics['convergence_speed']}")
    log.info("=" * 80)


if __name__ == "__main__":
    main()
