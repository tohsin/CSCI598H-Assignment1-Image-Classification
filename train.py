"""Train one of three classifiers on CIFAR-10 and optionally write CSV."""

import argparse
import csv
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from linear_classifier import compute_accuracy, train_one_step
from models import build_model


RESULT_FIELDS = ["experiment_id", "model", "optimizer", "epoch", "epochs",
                 "batch_size", "learning_rate", "seed", "train_loss",
                 "validation_accuracy", "epoch_time_seconds"]


def evaluate(model, data_loader, device):
    # TODO: Evaluate all batches without gradients. Weight each batch accuracy
    # by its number of examples and return the overall accuracy.
    raise NotImplementedError


def build_optimizer(name, parameters, learning_rate):
    if name.lower() == "sgd":
        return torch.optim.SGD(parameters, lr=learning_rate)
    if name.lower() == "adam":
        return torch.optim.Adam(parameters, lr=learning_rate)
    raise ValueError(f"Unknown optimizer {name!r}; choose sgd or adam.")


def make_data_loaders(data_dir, batch_size, seed, max_train_samples=None,
                      max_validation_samples=None):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])
    train_set = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform)
    validation_set = datasets.CIFAR10(data_dir, train=False, download=True, transform=transform)
    if max_train_samples is not None:
        train_set = Subset(train_set, range(min(max_train_samples, len(train_set))))
    if max_validation_samples is not None:
        validation_set = Subset(validation_set, range(min(max_validation_samples, len(validation_set))))
    generator = torch.Generator().manual_seed(seed)
    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True, generator=generator),
        DataLoader(validation_set, batch_size=batch_size, shuffle=False),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--experiment-id", default="single_run")
    parser.add_argument(
        "--model",
        choices=("linear", "three_layer", "conv", "vit"),
        default="linear",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--optimizer", choices=("sgd", "adam"), default="sgd")
    parser.add_argument("--seed", type=int, default=598)
    parser.add_argument("--output")
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-validation-samples", type=int)
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        raise ValueError("epochs, batch size, and learning rate must be positive")

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, validation_loader = make_data_loaders(
        args.data_dir, args.batch_size, args.seed,
        args.max_train_samples, args.max_validation_samples,
    )

    model = build_model(args.model).to(device)
    optimizer = build_optimizer(args.optimizer, model.parameters(), args.learning_rate)
    rows = []

    for epoch in range(1, args.epochs + 1):
        start = time.perf_counter()
        model.train()
        total_loss = 0.0
        total_examples = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            count = labels.size(0)
            total_loss += train_one_step(model, images, labels, optimizer) * count
            total_examples += count

        validation_accuracy = evaluate(model, validation_loader, device)
        mean_loss = total_loss / total_examples
        elapsed = time.perf_counter() - start
        rows.append({"experiment_id": args.experiment_id, "model": args.model,
                     "optimizer": args.optimizer, "epoch": epoch, "epochs": args.epochs,
                     "batch_size": args.batch_size, "learning_rate": args.learning_rate,
                     "seed": args.seed, "train_loss": f"{mean_loss:.8f}",
                     "validation_accuracy": f"{validation_accuracy:.8f}",
                     "epoch_time_seconds": f"{elapsed:.4f}"})
        print(
            f"Epoch {epoch:02d} | "
            f"loss={mean_loss:.4f} | validation_accuracy={validation_accuracy:.2%}"
        )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=RESULT_FIELDS)
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
