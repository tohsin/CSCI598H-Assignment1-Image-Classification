# CSCI 598H: CIFAR-10 Classifier Comparison

This is an individual assignment repository. Keep it private, commit your work
regularly, and push the final `results.csv` and `analysis.md` before selecting
the repository in Gradescope. See [SUBMISSION.md](SUBMISSION.md) for the final
checklist.

## Goal

Implement and compare three PyTorch classifiers on CIFAR-10:

1. A single linear layer.
2. A three-layer fully connected network.
3. A small convolutional neural network (CNN).
4. A compact Vision Transformer (ViT) with configurable encoder depth.

You will also compare epochs, batch size, learning rate, and optimizer. The
experiment configuration and results use machine-readable files so Gradescope
can check that the comparison is complete and internally consistent.

## Required implementation

Complete every `TODO` in:

- `linear_classifier.py`: forward pass, stable cross-entropy, one training
  step, and accuracy.
- `models.py`: `ThreeLayerClassifier`, `ConvClassifier`, and
  `VisionTransformerClassifier`.
- `train.py`: `evaluate`.

Follow the exact CNN architecture documented in `models.py`. All classifiers
must return logits, not softmax probabilities. Do not use
`torch.nn.functional.cross_entropy` in `softmax_cross_entropy`.

## Run one model

Install dependencies and run a short check:

```bash
python -m pip install -r requirements.txt
python train.py --model linear --epochs 1 --max-train-samples 2000 --max-validation-samples 500
python train.py --model three_layer --optimizer adam --learning-rate 0.001 --epochs 1 --max-train-samples 2000 --max-validation-samples 500
python train.py --model conv --num-layers 2 --optimizer adam --learning-rate 0.001 --epochs 1 --max-train-samples 2000 --max-validation-samples 500
python train.py --model vit --num-layers 2 --optimizer adam --learning-rate 0.001 --epochs 1 --max-train-samples 2000 --max-validation-samples 500
```

The `--max-*-samples` flags are intended for quick local checks. Do not add
them when producing the final experiment results.

## Run the comparison

`experiments.json` contains one linear baseline, one three-layer baseline, and
one-variable-at-a-time comparisons for CNN and ViT. These comparisons vary
epochs, batch size, learning rate, optimizer, and number of layers. For CNN,
`num_layers` means convolutional layers; for ViT, it means Transformer encoder
layers. The required values are 2 and 5. You may add experiments, but do not
remove or rename the provided experiment IDs.

```bash
python run_experiments.py --config experiments.json --output results.csv
```

Each CSV row records one epoch. Do not edit generated measurements by hand.
Complete the questions in `analysis.md` after the run. CIFAR-10's official
test split is called the validation set in this assignment because it is used
for model comparison; do not interpret these results as an untouched final
test estimate.

## Submission

Push these files to the root of your assigned private GitHub repository:

```text
linear_classifier.py
models.py
train.py
run_experiments.py
experiments.json
results.csv
analysis.md
```

Do not include the downloaded `data` directory. GitHub Actions runs public
synthetic-data smoke tests after each push. Gradescope uses additional synthetic data
for code tests and does not download CIFAR-10. It checks model architecture,
training behavior, optimizer selection, experiment coverage, and CSV
consistency. Submit the repository through Gradescope's GitHub integration;
written analysis is reviewed separately according to the course rubric.

Follow the course policy on collaboration and AI tools.
