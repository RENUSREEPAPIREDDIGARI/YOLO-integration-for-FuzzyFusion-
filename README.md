# Standalone YOLO Detection Starter

This repository is a separate object-detection project. It does not import,
modify, or depend on the Fuzzy-Guided Dynamic Feature Fusion project.

The starter is configured for one example class (`polyp`). It includes no
training data or trained weights. You need images with bounding-box
annotations before training; image-level class labels alone are not sufficient
for object detection.

## Setup

```bash
python -m venv .venv
```

Activate the environment, then install dependencies:

```bash
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate

python -m pip install -r requirements.txt
```

## Prepare annotated data

Follow [data/README.md](data/README.md) to create the YOLO image/label folder
structure. Update [data/colorectal.yaml](data/colorectal.yaml) with the actual
class names. The default split paths are relative to that YAML file.

## Train

From the repository root:

```bash
python scripts/train.py
```

Training settings are in [config.yaml](config.yaml). Override the dataset
configuration or model with:

```bash
python scripts/train.py --data path/to/dataset.yaml --model yolo11n.pt
```

Ultralytics downloads the selected pretrained model on first use. Training
outputs are saved under `runs/detect/` and are excluded from Git.

## Predict

After training:

```bash
python scripts/predict.py path/to/image.jpg
```

Predictions are saved under `runs/predict/`. To use another trained checkpoint:

```bash
python scripts/predict.py path/to/image.jpg --weights path/to/best.pt
```

## Data and privacy

Dataset files, labels, model weights, and experiment outputs are ignored by
Git. Do not publish medical images or annotations unless you have permission
and they are appropriately de-identified.
