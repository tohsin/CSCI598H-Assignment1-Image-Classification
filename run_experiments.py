"""Run every experiment in a JSON manifest and combine its CSV rows."""

import argparse
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path


FIELDS = ["experiment_id", "model", "optimizer", "epoch", "epochs",
          "batch_size", "learning_rate", "num_layers", "seed", "train_loss",
          "validation_accuracy", "epoch_time_seconds"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments.json")
    parser.add_argument("--output", default="results.csv")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-validation-samples", type=int)
    args = parser.parse_args()
    experiments = json.loads(Path(args.config).read_text()).get("experiments", [])
    if not experiments:
        raise ValueError("The manifest must contain at least one experiment.")

    required = {"id", "model", "epochs", "batch_size", "learning_rate", "optimizer", "seed"}
    all_rows = []
    with tempfile.TemporaryDirectory() as temp_dir:
        for index, item in enumerate(experiments):
            missing = required - item.keys()
            if missing:
                raise ValueError(f"Experiment is missing fields: {sorted(missing)}")
            if item["model"] in {"conv", "vit"} and item.get("num_layers") not in {2, 5}:
                raise ValueError(
                    f"{item['id']} must set num_layers to either 2 or 5"
                )
            result_path = Path(temp_dir) / f"result_{index}.csv"
            command = [sys.executable, "train.py", "--data-dir", args.data_dir,
                       "--experiment-id", str(item["id"]), "--model", str(item["model"]),
                       "--epochs", str(item["epochs"]), "--batch-size", str(item["batch_size"]),
                       "--learning-rate", str(item["learning_rate"]),
                       "--optimizer", str(item["optimizer"]), "--seed", str(item["seed"]),
                       "--output", str(result_path)]
            if item.get("num_layers") is not None:
                command += ["--num-layers", str(item["num_layers"])]
            if args.max_train_samples:
                command += ["--max-train-samples", str(args.max_train_samples)]
            if args.max_validation_samples:
                command += ["--max-validation-samples", str(args.max_validation_samples)]
            subprocess.run(command, check=True)
            with result_path.open(newline="") as result_file:
                all_rows.extend(csv.DictReader(result_file))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Wrote {len(all_rows)} epoch records to {output_path}")


if __name__ == "__main__":
    main()
