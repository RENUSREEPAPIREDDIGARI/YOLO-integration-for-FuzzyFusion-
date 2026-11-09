from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the standalone YOLO detector.")
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--data", type=Path, help="Override the dataset YAML path.")
    parser.add_argument("--model", help="Override the model checkpoint or model name.")
    args = parser.parse_args()

    config = load_config(args.config.resolve())
    data_path = args.data or Path(str(config["data"]))
    if not data_path.is_absolute():
        data_path = ROOT / data_path
    data_path = data_path.resolve()
    if not data_path.is_file():
        raise FileNotFoundError(
            f"Dataset YAML not found: {data_path}. Add annotated data using data/README.md."
        )

    model_name = args.model or str(config["model"])
    device = config.get("device", "auto")
    if device == "auto":
        device = None

    model = YOLO(model_name)
    model.train(
        data=str(data_path),
        epochs=int(config.get("epochs", 100)),
        imgsz=int(config.get("imgsz", 640)),
        batch=config.get("batch", 16),
        device=device,
        project=str(ROOT / str(config.get("project", "runs/detect"))),
        name=str(config.get("name", "colorectal-baseline")),
    )


if __name__ == "__main__":
    main()
