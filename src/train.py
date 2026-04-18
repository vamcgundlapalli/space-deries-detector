"""
src/train.py
────────────────────────────────────────────────────────────────────────────
YOLOv8 Training Script for Space Debris Detection
────────────────────────────────────────────────────────────────────────────
Usage:
    python src/train.py [--epochs 100] [--imgsz 640] [--batch 16] [--device cpu]
────────────────────────────────────────────────────────────────────────────
"""

import argparse
import os
import sys
from pathlib import Path

from ultralytics import YOLO


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent          # project root
DATASET_YAML = ROOT / "dataset" / "dataset.yaml"
MODELS_DIR = ROOT / "models"
PRETRAINED_WEIGHTS = "yolov8n.pt"                      # nano – fast baseline


# ─────────────────────────────────────────────────────────────────────────────
# Argument Parser
# ─────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 model for space debris detection"
    )
    parser.add_argument("--weights",  type=str,   default=PRETRAINED_WEIGHTS,
                        help="Pre-trained weights (e.g. yolov8n.pt, yolov8s.pt)")
    parser.add_argument("--data",     type=str,   default=str(DATASET_YAML),
                        help="Path to dataset YAML file")
    parser.add_argument("--epochs",   type=int,   default=100,
                        help="Number of training epochs")
    parser.add_argument("--imgsz",    type=int,   default=640,
                        help="Input image size (pixels)")
    parser.add_argument("--batch",    type=int,   default=16,
                        help="Batch size (-1 = AutoBatch)")
    parser.add_argument("--device",   type=str,   default="",
                        help="Device: '' = auto, 'cpu', '0', '0,1', etc.")
    parser.add_argument("--patience", type=int,   default=20,
                        help="Early-stopping patience (epochs without improvement)")
    parser.add_argument("--workers",  type=int,   default=4,
                        help="Dataloader worker threads")
    parser.add_argument("--project",  type=str,   default=str(MODELS_DIR),
                        help="Directory to save training runs")
    parser.add_argument("--name",     type=str,   default="space_debris_yolov8",
                        help="Run name (sub-folder inside --project)")
    parser.add_argument("--resume",   action="store_true",
                        help="Resume interrupted training")
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Validation helpers
# ─────────────────────────────────────────────────────────────────────────────
def validate_dataset(yaml_path: str) -> None:
    """Confirm dataset YAML and at least the train split exist."""
    p = Path(yaml_path)
    if not p.exists():
        sys.exit(
            f"[ERROR] Dataset YAML not found: {yaml_path}\n"
            "Run  python scripts/download_dataset.py  to generate a sample dataset."
        )
    # Quick sanity-check: look for training images
    import yaml  # PyYAML is installed with ultralytics
    with open(p) as f:
        cfg = yaml.safe_load(f)

    dataset_root = Path(cfg.get("path", p.parent))
    train_dir = dataset_root / cfg.get("train", "images/train")
    if not train_dir.exists() or not any(train_dir.iterdir()):
        sys.exit(
            f"[ERROR] No training images found in {train_dir}\n"
            "Run  python scripts/download_dataset.py  first."
        )
    print(f"[INFO] Dataset OK  →  {yaml_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Core training function
# ─────────────────────────────────────────────────────────────────────────────
def train(args: argparse.Namespace) -> Path:
    """
    Load a pre-trained YOLOv8 model and fine-tune it on the debris dataset.

    Returns:
        Path to the best saved weights.
    """
    validate_dataset(args.data)

    print(f"\n{'─'*60}")
    print("  Space Debris Detection — YOLOv8 Training")
    print(f"{'─'*60}")
    print(f"  Weights  : {args.weights}")
    print(f"  Dataset  : {args.data}")
    print(f"  Epochs   : {args.epochs}")
    print(f"  Img size : {args.imgsz}px")
    print(f"  Batch    : {args.batch}")
    print(f"  Device   : {args.device or 'auto'}")
    print(f"{'─'*60}\n")

    # Load model from Ultralytics hub or local path
    model = YOLO(args.weights)

    # Kick off training
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device if args.device else None,
        patience=args.patience,
        workers=args.workers,
        project=args.project,
        name=args.name,
        resume=args.resume,
        # Augmentation settings (good defaults for small datasets)
        hsv_h=0.015,
        hsv_s=0.4,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        flipud=0.3,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        # Optimizer
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=3,
        # Logging
        plots=True,
        save=True,
        save_period=10,
    )

    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    print(f"\n[SUCCESS] Training complete!")
    print(f"  Best weights → {best_weights}")
    return best_weights


# ─────────────────────────────────────────────────────────────────────────────
# Post-training evaluation
# ─────────────────────────────────────────────────────────────────────────────
def evaluate(weights_path: Path, data_yaml: str) -> None:
    """Run validation on the best checkpoint and print metrics."""
    print(f"\n[INFO] Evaluating best model on validation set …")
    model = YOLO(str(weights_path))
    metrics = model.val(data=data_yaml)
    print(f"\n{'─'*60}")
    print("  Validation Metrics")
    print(f"{'─'*60}")
    print(f"  mAP@0.5        : {metrics.box.map50:.4f}")
    print(f"  mAP@0.5:0.95   : {metrics.box.map:.4f}")
    print(f"  Precision       : {metrics.box.mp:.4f}")
    print(f"  Recall          : {metrics.box.mr:.4f}")
    print(f"{'─'*60}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    best_weights = train(args)

    if best_weights.exists():
        evaluate(best_weights, args.data)
    else:
        print("[WARN] best.pt not found – skipping evaluation.")
