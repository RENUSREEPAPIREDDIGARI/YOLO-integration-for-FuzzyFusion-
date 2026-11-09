# Standalone YOLO Image Classification

This repository is a separate image-classification project. It does not import
or modify the Fuzzy-Guided Dynamic Feature Fusion project.

The fuzzy project's dataset is organized into class folders, so it can be
reused to train a **YOLO classification** model. It does not include
bounding-box annotations, so it cannot be used for object detection without
additional labeling.

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

## Train using the fuzzy project's dataset

By default, [config.yaml](config.yaml) points to the sibling
`CI Project/kvasir-dataset` directory. From this repository's root:

```bash
python scripts/train.py
```

To use a dataset in another location:

```bash
python scripts/train.py --source "path/to/class-folder-dataset"
```

The source must contain one subfolder per class, with images inside each
subfolder. The script makes a deterministic 70/15/15 train/validation/test
split (controlled by `seed` in [config.yaml](config.yaml)) and copies it into
the ignored `data/colorectal-cls/` folder. It never writes to the source
dataset. See [data/README.md](data/README.md) for details.

Training settings are in [config.yaml](config.yaml). To use another output
folder or model:

```bash
python scripts/train.py --source "path/to/dataset" --data "data/my-split" --model yolo11n-cls.pt
```

Ultralytics downloads the selected pretrained model on first use. Training
outputs are saved under `runs/classify/`; model weights and run output are
excluded from Git.

## Predict

After training:

```bash
python scripts/predict.py path/to/image.jpg
```

Predictions are saved under `runs/classify-predict/`. To use another trained
checkpoint:

```bash
python scripts/predict.py path/to/image.jpg --weights path/to/best.pt
```

## Data and privacy

Dataset files, prepared splits, model weights, and experiment outputs are
ignored by Git. Do not publish medical images or results unless you have
permission and they are appropriately de-identified.
