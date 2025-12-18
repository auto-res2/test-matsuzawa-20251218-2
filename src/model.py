
"""
Complete model architecture implementations for all experiments.
Provides ResNet-20 (CIFAR-10), ResNet-50 (ImageNet), and BERT models.
"""

import logging
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)


# ============================================================================
# ResNet Components (CIFAR-10 and ImageNet)
# ============================================================================

class BasicBlock(nn.Module):
    """Basic residual block for ResNet."""
    expansion = 1
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1,
                 downsample: Optional[nn.Module] = None):
        """Initialize BasicBlock."""
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
        """Forward pass."""
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = F.relu(out, inplace=True)
        
        return out


class Bottleneck(nn.Module):
    """Bottleneck residual block for ResNet (used in ResNet-50+)."""
    expansion = 4
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1,
                 downsample: Optional[nn.Module] = None):
        """Initialize Bottleneck."""
        super(Bottleneck, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion,
                              kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out, inplace=True)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = F.relu(out, inplace=True)
        
        return out


class ResNetCIFAR(nn.Module):
    """ResNet optimized for CIFAR-10/100 (20, 32, 44, 56, 110 layer variants)."""
    
    def __init__(self, block, num_blocks: List[int], num_classes: int = 10):
        """Initialize ResNetCIFAR."""
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
        """Create a residual layer."""
        
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
        """Initialize model weights."""
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
        """Forward pass."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x


class ResNetImageNet(nn.Module):
    """ResNet for ImageNet (standard architecture)."""
    
    def __init__(self, block, num_blocks: List[int], num_classes: int = 1000,
                 in_channels: int = 3, initial_channels: int = 64):
        """Initialize ResNetImageNet."""
        super(ResNetImageNet, self).__init__()
        
        self.in_channels = initial_channels
        
        self.conv1 = nn.Conv2d(in_channels, initial_channels, kernel_size=7,
                              stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(initial_channels)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        self._init_weights()
    
    def _make_layer(self, block, out_channels: int, num_blocks: int,
                   stride: int = 1) -> nn.Sequential:
        """Create a residual layer."""
        
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
        """Initialize model weights."""
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
        assert self.fc.out_features == 1000, \
            f"FC layer output dimension should be 1000, got {self.fc.out_features}"
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x


# ============================================================================
# BERT Wrapper for NLP Tasks
# ============================================================================

class BertModelWrapper(nn.Module):
    """Wrapper for HuggingFace BERT model for fine-tuning tasks."""
    
    def __init__(self, model_name: str = "bert-base-uncased", num_classes: int = 2,
                 task_type: str = "classification", cache_dir: str = ".cache"):
        """Initialize BERT wrapper."""
        super(BertModelWrapper, self).__init__()
        
        try:
            from transformers import AutoModel, AutoConfig
        except ImportError:
            raise ImportError("transformers library required for BERT models")
        
        self.model_name = model_name
        self.num_classes = num_classes
        self.task_type = task_type
        
        # Load base model
        self.config = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir)
        self.bert = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
        
        # Task-specific heads
        if task_type == "classification":
            self.classifier = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(self.config.hidden_size, num_classes)
            )
        elif task_type == "qa":
            # SQuAD-style QA: predict start and end positions
            self.qa_outputs = nn.Linear(self.config.hidden_size, 2)
        else:
            raise ValueError(f"Unknown task_type: {task_type}")
        
        log.info(f"Created BERT wrapper: {model_name}, task={task_type}, classes={num_classes}")
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
               token_type_ids: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        """Forward pass."""
        
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )
        
        if self.task_type == "classification":
            pooled_output = outputs.pooler_output
            logits = self.classifier(pooled_output)
            return logits
        elif self.task_type == "qa":
            sequence_output = outputs.last_hidden_state
            logits = self.qa_outputs(sequence_output)
            start_logits, end_logits = logits.split(1, dim=-1)
            return start_logits.squeeze(-1), end_logits.squeeze(-1)
        
        return outputs


# ============================================================================
# Factory Functions
# ============================================================================

def resnet20_cifar(num_classes: int = 10) -> ResNetCIFAR:
    """ResNet-20 for CIFAR-10/100."""
    return ResNetCIFAR(BasicBlock, [3, 3, 3], num_classes=num_classes)


def resnet50_imagenet(num_classes: int = 1000) -> ResNetImageNet:
    """ResNet-50 for ImageNet."""
    return ResNetImageNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)


def bert_base_classification(num_classes: int = 2, cache_dir: str = ".cache") -> BertModelWrapper:
    """BERT-base for classification tasks."""
    return BertModelWrapper(
        model_name="bert-base-uncased",
        num_classes=num_classes,
        task_type="classification",
        cache_dir=cache_dir
    )


def bert_base_qa(cache_dir: str = ".cache") -> BertModelWrapper:
    """BERT-base for question answering (SQuAD-style)."""
    return BertModelWrapper(
        model_name="bert-base-uncased",
        num_classes=2,
        task_type="qa",
        cache_dir=cache_dir
    )


# ============================================================================
# Model Registry
# ============================================================================

MODEL_REGISTRY = {
    'resnet20_cifar': resnet20_cifar,
    'resnet20': resnet20_cifar,
    'resnet50': resnet50_imagenet,
    'resnet50_imagenet': resnet50_imagenet,
    'bert_base': bert_base_classification,
    'bert_base_qa': bert_base_qa,
}


def create_model(model_name: str, num_classes: int = 10, task_type: str = "classification",
                cache_dir: str = ".cache") -> nn.Module:
    """
    Create model by name from registry.
    
    Args:
        model_name: Model identifier
        num_classes: Number of output classes (for classification)
        task_type: Task type (for BERT models)
        cache_dir: Cache directory for model downloads
    
    Returns:
        Model instance
    """
    
    model_name_lower = model_name.lower()
    
    if model_name_lower not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}")
    
    model_fn = MODEL_REGISTRY[model_name_lower]
    
    # Call appropriate factory based on model type
    if 'bert' in model_name_lower:
        if 'qa' in model_name_lower:
            model = model_fn(cache_dir=cache_dir)
        else:
            model = model_fn(num_classes=num_classes, cache_dir=cache_dir)
    else:
        model = model_fn(num_classes=num_classes)
    
    log.info(f"Created model: {model_name} with {num_classes} classes")
    
    return model


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test model creation
    models = [
        ('resnet20_cifar', 10),
        ('resnet50', 1000),
    ]
    
    for model_name, num_classes in models:
        try:
            model = create_model(model_name, num_classes=num_classes)
            total_params = sum(p.numel() for p in model.parameters())
            log.info(f"{model_name}: {total_params:,} parameters")
        except Exception as e:
            log.error(f"Failed to create {model_name}: {e}")
