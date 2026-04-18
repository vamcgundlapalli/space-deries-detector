"""
src/detect.py
────────────────────────────────────────────────────────────────────────────
Inference / Detection Script for Space Debris Detection
────────────────────────────────────────────────────────────────────────────
Supports:
    • Single image file
    • Directory of images
    • Live webcam stream

Usage:
    python src/detect.py --source path/to/image.jpg
    python src/detect.py --source path/to/images/
    python src/detect.py --source 0              # webcam index 0
    python src/detect.py --source 0 --webcam
────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from preprocess import preprocess_for_inference


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = ROOT / "models" / "space_debris_yolov8" / "weights" / "best.pt"
OUTPUT_DIR = ROOT / "runs" / "detect"

# Class colours (BGR)  –  debris=red, defunct_satellite=cyan, rocket_body=magenta
CLASS_COLORS: Dict[int, Tuple[int, int, int]] = {
    0: (0,   50, 255),   # debris           – vivid red-orange
    1: (255, 200,  0),   # defunct_satellite – golden-yellow
    2: (200,   0, 255),  # rocket_body      – purple
}
DEFAULT_COLOR = (0, 255, 128)   # fallback green for unknown class ids


# ─────────────────────────────────────────────────────────────────────────────
# Dataclass for a single detection result
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Detection:
    bbox: Tuple[int, int, int, int]   # x1, y1, x2, y2 (pixel coords)
    class_id: int
    class_name: str
    confidence: float


@dataclass
class FrameResult:
    image: np.ndarray                 # annotated BGR image
    detections: List[Detection] = field(default_factory=list)
    inference_ms: float = 0.0

    @property
    def count(self) -> int:
        return len(self.detections)

    def counts_by_class(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for d in self.detections:
            counts[d.class_name] = counts.get(d.class_name, 0) + 1
        return counts


# ─────────────────────────────────────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect space debris using a trained YOLOv8 model."
    )
    parser.add_argument("--weights",    type=str, default=str(DEFAULT_WEIGHTS),
                        help="Path to trained .pt weights")
    parser.add_argument("--source",     type=str, required=True,
                        help="Image path, image directory, or webcam index (e.g. 0)")
    parser.add_argument("--conf",       type=float, default=0.25,
                        help="Confidence threshold [0–1]")
    parser.add_argument("--iou",        type=float, default=0.45,
                        help="NMS IoU threshold [0–1]")
    parser.add_argument("--imgsz",      type=int,   default=640,
                        help="Inference image size")
    parser.add_argument("--device",     type=str,   default="",
                        help="Device: '' = auto, 'cpu', '0', etc.")
    parser.add_argument("--save",       action="store_true",
                        help="Save annotated output to disk")
    parser.add_argument("--show",       action="store_true",
                        help="Display annotated results in an OpenCV window")
    parser.add_argument("--webcam",     action="store_true",
                        help="Force webcam mode (source must be integer index)")
    parser.add_argument("--denoise",    action="store_true",
                        help="Apply denoising before inference")
    parser.add_argument("--enhance",    action="store_true",
                        help="Apply CLAHE contrast enhancement before inference")
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Model loader
# ─────────────────────────────────────────────────────────────────────────────
def load_model(weights: str, device: str = "") -> YOLO:
    """Load YOLOv8 model from *weights* path."""
    wp = Path(weights)
    if not wp.exists():
        sys.exit(
            f"[ERROR] Weights not found: {weights}\n"
            "Train the model first:  python src/train.py"
        )
    print(f"[INFO] Loading model: {wp}")
    model = YOLO(str(wp))
    if device:
        model.to(device)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Annotation helpers
# ─────────────────────────────────────────────────────────────────────────────
def _draw_box(
    image: np.ndarray,
    det: Detection,
    font_scale: float = 0.55,
    thickness: int = 2,
) -> None:
    """Draw a bounding box + label on *image* in-place."""
    x1, y1, x2, y2 = det.bbox
    color = CLASS_COLORS.get(det.class_id, DEFAULT_COLOR)
    label = f"{det.class_name} {det.confidence:.2f}"

    # Box
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

    # Label background
    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                          font_scale, thickness)
    label_y = max(y1, th + baseline + 4)
    cv2.rectangle(image,
                  (x1, label_y - th - baseline - 4),
                  (x1 + tw + 4, label_y),
                  color, cv2.FILLED)
    # Label text
    cv2.putText(image, label,
                (x1 + 2, label_y - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                (255, 255, 255), thickness, cv2.LINE_AA)


def _draw_hud(image: np.ndarray, result: FrameResult) -> None:
    """Overlay detection count + per-class breakdown at top-left."""
    overlay_lines = [f"  Total debris: {result.count}"]
    for cls, cnt in result.counts_by_class().items():
        overlay_lines.append(f"    {cls}: {cnt}")
    overlay_lines.append(f"  Inference: {result.inference_ms:.1f} ms")

    y0, dy = 24, 22
    for i, line in enumerate(overlay_lines):
        y = y0 + i * dy
        # Shadow
        cv2.putText(image, line, (11, y + 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
        # Text
        cv2.putText(image, line, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 200), 2, cv2.LINE_AA)


def annotate_frame(image: np.ndarray, result: FrameResult) -> np.ndarray:
    """Return a copy of *image* with all detections drawn."""
    annotated = image.copy()
    for det in result.detections:
        _draw_box(annotated, det)
    _draw_hud(annotated, result)
    return annotated


# ─────────────────────────────────────────────────────────────────────────────
# Core inference
# ─────────────────────────────────────────────────────────────────────────────
def run_inference(
    model: YOLO,
    image: np.ndarray,
    conf: float = 0.25,
    iou: float = 0.45,
    imgsz: int = 640,
    denoise: bool = False,
    enhance: bool = False,
) -> FrameResult:
    """
    Run YOLOv8 inference on a single BGR frame.

    Args:
        model:   Loaded YOLO model.
        image:   uint8 BGR numpy array.
        conf:    Confidence threshold.
        iou:     NMS IoU threshold.
        imgsz:   Inference resolution.
        denoise: Pre-denoise image.
        enhance: Pre-enhance contrast.

    Returns:
        FrameResult with detections and annotated image.
    """
    # Optional preprocessing
    preprocessed = preprocess_for_inference(
        image,
        target_size=imgsz,
        denoise=denoise,
        enhance=enhance,
    )

    t0 = time.perf_counter()
    results = model.predict(
        source=preprocessed,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        verbose=False,
    )
    inference_ms = (time.perf_counter() - t0) * 1000

    detections: List[Detection] = []
    names = model.names  # {id: class_name}

    for r in results:
        if r.boxes is None:
            continue
        boxes = r.boxes.xyxy.cpu().numpy().astype(int)
        confs = r.boxes.conf.cpu().numpy()
        cls_ids = r.boxes.cls.cpu().numpy().astype(int)

        for (x1, y1, x2, y2), cf, cid in zip(boxes, confs, cls_ids):
            detections.append(Detection(
                bbox=(x1, y1, x2, y2),
                class_id=int(cid),
                class_name=names.get(int(cid), f"class_{cid}"),
                confidence=float(cf),
            ))

    frame_result = FrameResult(
        image=image,
        detections=detections,
        inference_ms=inference_ms,
    )
    frame_result.image = annotate_frame(image, frame_result)
    return frame_result


# ─────────────────────────────────────────────────────────────────────────────
# Source handlers
# ─────────────────────────────────────────────────────────────────────────────
def _is_webcam_source(source: str) -> bool:
    try:
        int(source)
        return True
    except ValueError:
        return False


def detect_image(
    model: YOLO,
    image_path: Path,
    conf: float,
    iou: float,
    imgsz: int,
    denoise: bool,
    enhance: bool,
    save: bool,
    show: bool,
) -> FrameResult:
    """Detect debris in a single image file."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    result = run_inference(model, img, conf, iou, imgsz, denoise, enhance)

    print(f"  {image_path.name}  →  {result.count} detection(s) "
          f"({result.inference_ms:.1f} ms)")
    for d in result.detections:
        print(f"    [{d.class_name}] conf={d.confidence:.3f}  "
              f"bbox={d.bbox}")

    if save:
        out_path = OUTPUT_DIR / image_path.name
        cv2.imwrite(str(out_path), result.image)
        print(f"  Saved → {out_path}")

    if show:
        cv2.imshow(f"Space Debris — {image_path.name}", result.image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return result


def detect_directory(
    model: YOLO,
    dir_path: Path,
    conf: float,
    iou: float,
    imgsz: int,
    denoise: bool,
    enhance: bool,
    save: bool,
    show: bool,
) -> List[FrameResult]:
    """Run detection on every image inside *dir_path*."""
    EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    images = sorted(p for p in dir_path.iterdir() if p.suffix.lower() in EXTS)
    if not images:
        print(f"[WARN] No images found in {dir_path}")
        return []

    results = []
    for img_path in images:
        r = detect_image(model, img_path, conf, iou, imgsz,
                         denoise, enhance, save, show)
        results.append(r)
    return results


def detect_webcam(
    model: YOLO,
    cam_index: int,
    conf: float,
    iou: float,
    imgsz: int,
    denoise: bool,
    enhance: bool,
    save: bool,
) -> None:
    """
    Real-time debris detection from a webcam stream.
    Press  Q  or  ESC  to quit.
    Press  S  to save the current annotated frame.
    """
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        sys.exit(f"[ERROR] Cannot open webcam index {cam_index}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("[INFO] Webcam stream started — press Q/ESC to quit, S to save frame")

    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to grab frame – retrying …")
            time.sleep(0.05)
            continue

        result = run_inference(model, frame, conf, iou, imgsz, denoise, enhance)

        # FPS overlay
        fps_label = f"FPS: {1000 / max(result.inference_ms, 1):.1f}"
        cv2.putText(result.image, fps_label,
                    (result.image.shape[1] - 140, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2, cv2.LINE_AA)

        cv2.imshow("Space Debris — Real-time Detection (Q to quit)", result.image)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):   # Q or ESC
            break
        if key in (ord("s"), ord("S")):
            fname = OUTPUT_DIR / f"webcam_frame_{frame_id:05d}.jpg"
            cv2.imwrite(str(fname), result.image)
            print(f"  Saved frame → {fname}")

        frame_id += 1

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Webcam stream closed.")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = load_model(args.weights, args.device)

    source = args.source.strip()

    if args.webcam or _is_webcam_source(source):
        cam_idx = int(source)
        detect_webcam(model, cam_idx, args.conf, args.iou, args.imgsz,
                      args.denoise, args.enhance, args.save)
    else:
        src_path = Path(source)
        if not src_path.exists():
            sys.exit(f"[ERROR] Source not found: {src_path}")

        if src_path.is_dir():
            detect_directory(model, src_path, args.conf, args.iou, args.imgsz,
                             args.denoise, args.enhance, args.save, args.show)
        else:
            detect_image(model, src_path, args.conf, args.iou, args.imgsz,
                         args.denoise, args.enhance, args.save, args.show)


if __name__ == "__main__":
    main()
