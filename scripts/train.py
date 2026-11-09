from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml
from ultralytics import YOLO

from prepare_classification_data import prepare_classification_data

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare class-folder images and train a YOLO classifier."
    )
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--source", type=Path, help="Override the source class-folder dataset.")
    parser.add_argument("--data", type=Path, help="Override the prepared dataset output folder.")
    parser.add_argument("--model", help="Override the model checkpoint or model name.")
    args = parser.parse_args()

    config = load_config(args.config.resolve())
    source_path = args.source or Path(str(config["source"]))
    if not source_path.is_absolute():
        source_path = (ROOT / source_path).resolve()
    data_path = args.data or Path(str(config["data"]))
    if not data_path.is_absolute():
        data_path = ROOT / data_path
    data_path = data_path.resolve()
    prepare_classification_data(
        source=source_path,
        output=data_path,
        train_ratio=float(config.get("train_ratio", 0.7)),
        val_ratio=float(config.get("val_ratio", 0.15)),
        seed=int(config.get("seed", 42)),
    )

    model_name = args.model or str(config["model"])
    device = config.get("device", "auto")
    if device == "auto":
        device = None

    model = YOLO(model_name)
    model.train(
        data=str(data_path),
        epochs=int(config.get("epochs", 100)),
        imgsz=int(config.get("imgsz", 224)),
        batch=config.get("batch", 16),
        device=device,
        project=str(ROOT / str(config.get("project", "runs/classify"))),
        name=str(config.get("name", "colorectal-baseline")),
    )


if __name__ == "__main__":
    main()
