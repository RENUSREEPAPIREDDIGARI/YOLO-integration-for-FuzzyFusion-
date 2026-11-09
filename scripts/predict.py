from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the standalone YOLO detector.")
    parser.add_argument("source", type=Path, help="Image or directory to predict on.")
    parser.add_argument(
        "--weights",
        type=Path,
        default=ROOT / "runs" / "detect" / "colorectal-baseline" / "weights" / "best.pt",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    source = args.source.resolve()
    weights = args.weights.resolve()
    if not source.exists():
        raise FileNotFoundError(f"Prediction source not found: {source}")
    if not weights.is_file():
        raise FileNotFoundError(
            f"Model weights not found: {weights}. Train a model first or pass --weights."
        )

    model = YOLO(str(weights))
    model.predict(
        source=str(source),
        imgsz=args.imgsz,
        conf=args.conf,
        project=str(ROOT / "runs" / "predict"),
        save=True,
    )


if __name__ == "__main__":
    main()
