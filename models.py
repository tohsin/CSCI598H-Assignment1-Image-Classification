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

    def __init__(self, num_classes=10, num_layers=2):
        super().__init__()
        if num_layers not in {2, 5}:
            raise ValueError("ConvClassifier num_layers must be 2 or 5")
        self.num_layers = num_layers
        # TODO: Build num_layers convolutional blocks. Each block contains
        # Conv2d(kernel_size=3, padding=1), ReLU, and MaxPool2d(2).
        # Use output channels [32, 64] for the 2-layer model and
        # [32, 64, 128, 128, 128] for the 5-layer model. After the blocks,
        # flatten and use Linear(flattened_dim, 128), ReLU, and
        # Linear(128, num_classes). For 32x32 inputs, flattened_dim is
        # channels[-1] * (32 // 2**num_layers) ** 2.
        raise NotImplementedError

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return logits with shape [batch_size, num_classes]."""
        # TODO: Apply the convolutional network and return logits.
        raise NotImplementedError


class VisionTransformerClassifier(nn.Module):
    """A compact Vision Transformer for 32 x 32 RGB images.

    Required architecture:
    - split each image into non-overlapping 4 x 4 patches;
    - embed every patch into 128 dimensions;
    - prepend one learnable classification token;
    - add learnable positional embeddings;
    - apply either two or five Transformer encoder layers, each with four
      attention heads and a 256-dimensional feed-forward block;
    - classify the final classification-token representation.
    """

    def __init__(
        self,
        image_size=32,
        patch_size=4,
        in_channels=3,
        embed_dim=128,
        num_heads=4,
        mlp_dim=256,
        num_layers=2,
        num_classes=10,
        dropout=0.1,
    ):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        if num_layers not in {2, 5}:
            raise ValueError("VisionTransformerClassifier num_layers must be 2 or 5")

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2

        # TODO: Define all components below.
        # 1. Use Conv2d(in_channels, embed_dim, kernel_size=patch_size,
        #    stride=patch_size) as the patch embedding.
        # 2. Define learnable cls_token [1, 1, embed_dim] and positional
        #    embedding [1, num_patches + 1, embed_dim] parameters.
        # 3. Create nn.TransformerEncoderLayer with d_model=embed_dim,
        #    nhead=num_heads, dim_feedforward=mlp_dim, dropout=dropout, and
        #    batch_first=True. Wrap it in nn.TransformerEncoder with
        #    num_layers encoder layers.
        # 4. Add a final LayerNorm and Linear(embed_dim, num_classes) head.
        raise NotImplementedError

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return logits with shape [batch_size, num_classes]."""
        # TODO: Check the input spatial size, patch-embed the images, flatten
        # the patch grid into a token sequence, prepend a copy of cls_token for
        # each item, add positional embeddings, run the encoder, and
        # classify the normalized class token. Return logits, not probabilities.
        raise NotImplementedError


def build_model(name: str, num_classes=10, num_layers=None) -> nn.Module:
    """Create one of the assignment's classifier architectures."""
    normalized = name.lower().replace("-", "_")
    if normalized == "linear":
        return LinearClassifier(num_classes=num_classes)
    if normalized in {"three_layer", "mlp"}:
        return ThreeLayerClassifier(num_classes=num_classes)
    if normalized in {"conv", "cnn"}:
        return ConvClassifier(num_classes=num_classes, num_layers=num_layers or 2)
    if normalized in {"vit", "vision_transformer"}:
        return VisionTransformerClassifier(
            num_classes=num_classes, num_layers=num_layers or 2
        )
    raise ValueError(
        f"Unknown model {name!r}; choose linear, three_layer, conv, or vit."
    )
