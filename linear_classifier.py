"""Student implementation for the CIFAR-10 linear-classifier assignment."""

import torch
from torch import nn


class LinearClassifier(nn.Module):
    """A single linear layer for image classification."""

    def __init__(self, input_dim: int = 3 * 32 * 32, num_classes: int = 10):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return logits with shape [batch_size, num_classes]."""
        # TODO 1: Flatten each image, but keep the batch dimension.
        # TODO 2: Pass the flattened images through self.linear.
        raise NotImplementedError


def softmax_cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Return the mean softmax cross-entropy loss.

    Args:
        logits: Tensor with shape [batch_size, num_classes].
        labels: Integer tensor with shape [batch_size].

    Do not use torch.nn.functional.cross_entropy in this function.
    """
    # TODO 1: Compute log probabilities using a numerically stable method.
    # TODO 2: Select the log probability of each correct class.
    # TODO 3: Return the mean negative log probability.
    raise NotImplementedError


def train_one_step(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    optimizer: torch.optim.Optimizer,
) -> float:
    """Run one training step and return the loss as a Python float."""
    # TODO: Set gradients to zero.
    # TODO: Compute logits and loss.
    # TODO: Run backpropagation.
    # TODO: Update the parameters with the optimizer.
    raise NotImplementedError


def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Return the fraction of correct predictions as a Python float."""
    # TODO: Choose the class with the largest logit and compute accuracy.
    raise NotImplementedError
