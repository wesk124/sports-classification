"""Model definitions and factory functions for ResNet-based classifier."""

from typing import Literal

import torch.nn as nn
import torchvision.models as models
from torchvision.models import (
    ResNet18_Weights,
    ResNet50_Weights,
    ResNet101_Weights,
)

ModelName = Literal["resnet18", "resnet50", "resnet101"]


def _get_resnet(
    name: ModelName,
    num_classes: int,
    pretrained: bool = True,
    freeze_backbone: bool = True,
) -> nn.Module:
    """Return a ResNet model with a replaced final FC layer."""

    if name == "resnet18":
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
    elif name == "resnet50":
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
    elif name == "resnet101":
        weights = ResNet101_Weights.DEFAULT if pretrained else None
        model = models.resnet101(weights=weights)
    else:
        raise ValueError(f"Unsupported model name: {name}")

    # Optionally freeze backbone
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Replace the final FC layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


def create_model(
    model_name: ModelName,
    num_classes: int,
    pretrained: bool = True,
    freeze_backbone: bool = True,
) -> nn.Module:
    """Factory function to create a ResNet model with custom classifier."""
    return _get_resnet(
        name=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
    )
