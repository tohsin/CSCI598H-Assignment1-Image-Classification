"""Classifier architectures used in the CIFAR-10 comparison."""

import torch
from torch import nn

from linear_classifier import LinearClassifier


class ThreeLayerClassifier(nn.Module):
    """A three-layer fully connected classifier (three trainable layers)."""

    def __init__(self, input_dim=3 * 32 * 32, hidden_dim=512, num_classes=10):
        super().__init__()
        # TODO: Define Linear(input_dim, hidden_dim), Linear(hidden_dim,
        # hidden_dim), and Linear(hidden_dim, num_classes), with ReLU after
        # each of the first two linear layers.
        raise NotImplementedError

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return logits with shape [batch_size, num_classes]."""
        # TODO: Flatten each image and apply the three-layer network.
        raise NotImplementedError


class ConvClassifier(nn.Module):
    """A small convolutional network for 32 x 32 RGB images."""

    def __init__(self, num_classes=10):
        super().__init__()
        # TODO: Build this architecture:
        # Conv2d(3,32,3,padding=1), ReLU, MaxPool2d(2),
        # Conv2d(32,64,3,padding=1), ReLU, MaxPool2d(2),
        # Flatten, Linear(64*8*8,128), ReLU, Linear(128,num_classes).
        raise NotImplementedError

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return logits with shape [batch_size, num_classes]."""
        # TODO: Apply the convolutional network and return logits.
        raise NotImplementedError


def build_model(name: str, num_classes=10) -> nn.Module:
    """Create one of the assignment's classifier architectures."""
    normalized = name.lower().replace("-", "_")
    if normalized == "linear":
        return LinearClassifier(num_classes=num_classes)
    if normalized in {"three_layer", "mlp"}:
        return ThreeLayerClassifier(num_classes=num_classes)
    if normalized in {"conv", "cnn"}:
        return ConvClassifier(num_classes=num_classes)
    raise ValueError(f"Unknown model {name!r}; choose linear, three_layer, or conv.")
