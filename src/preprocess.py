"""
src/preprocess.py
────────────────────────────────────────────────────────────────────────────
Image preprocessing utilities for Space Debris Detection
────────────────────────────────────────────────────────────────────────────
All functions are side-effect-free and return new numpy arrays so they can
be composed freely in both the training pipeline and Streamlit app.
────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Type aliases
# ─────────────────────────────────────────────────────────────────────────────
Image = np.ndarray   # H×W×C  uint8  BGR (OpenCV convention)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Resize
# ─────────────────────────────────────────────────────────────────────────────
def resize_image(
    image: Image,
    target_size: int = 640,
    keep_aspect: bool = True,
    pad_color: Tuple[int, int, int] = (114, 114, 114),
) -> Image:
    """
    Resize *image* so its longest edge equals *target_size*.

    If *keep_aspect* is True the image is letterboxed with *pad_color* so the
    output is exactly (target_size, target_size, C).  Otherwise a direct
    cv2.resize is performed.

    Args:
        image:       BGR numpy array.
        target_size: Output side length in pixels (default 640 for YOLOv8).
        keep_aspect: Preserve aspect ratio and pad.
        pad_color:   BGR tuple used for padding pixels.

    Returns:
        Resized / letterboxed image as uint8 BGR array.
    """
    h, w = image.shape[:2]

    if not keep_aspect:
        return cv2.resize(image, (target_size, target_size),
                          interpolation=cv2.INTER_LINEAR)

    # Compute uniform scale factor
    scale = target_size / max(h, w)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Create canvas and paste resized image centered
    canvas = np.full((target_size, target_size, image.shape[2]),
                     pad_color, dtype=np.uint8)
    top  = (target_size - new_h) // 2
    left = (target_size - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = resized
    return canvas


# ─────────────────────────────────────────────────────────────────────────────
# 2. Normalize
# ─────────────────────────────────────────────────────────────────────────────
def normalize_image(
    image: Image,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std:  Tuple[float, float, float] = (0.229, 0.224, 0.225),
    to_rgb: bool = True,
) -> np.ndarray:
    """
    Normalize a uint8 image to float32 in [0, 1] then apply channel-wise
    (mean, std) normalization (ImageNet defaults).

    Args:
        image:  uint8 BGR numpy array.
        mean:   Per-channel mean  (RGB order).
        std:    Per-channel std   (RGB order).
        to_rgb: Convert BGR → RGB before normalization.

    Returns:
        float32 array with shape (H, W, 3), normalized.
    """
    img = image.astype(np.float32) / 255.0
    if to_rgb:
        img = img[:, :, ::-1]   # BGR → RGB
    img = (img - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)
    return img


# ─────────────────────────────────────────────────────────────────────────────
# 3. Noise reduction
# ─────────────────────────────────────────────────────────────────────────────
def denoise_image(
    image: Image,
    method: str = "gaussian",
    kernel_size: int = 3,
    # Non-local-means params
    h: float = 10.0,
    template_window: int = 7,
    search_window: int = 21,
) -> Image:
    """
    Reduce sensor / compression noise.

    Args:
        image:           uint8 BGR image.
        method:          'gaussian' | 'median' | 'bilateral' | 'nlm'
                         (non-local means).
        kernel_size:     Kernel size for gaussian / median (must be odd).
        h:               Filter strength for NLM (higher = more smoothing).
        template_window: Template patch size for NLM.
        search_window:   Search window size for NLM.

    Returns:
        Denoised uint8 BGR image.
    """
    method = method.lower()
    ks = kernel_size if kernel_size % 2 == 1 else kernel_size + 1  # ensure odd

    if method == "gaussian":
        return cv2.GaussianBlur(image, (ks, ks), 0)
    elif method == "median":
        return cv2.medianBlur(image, ks)
    elif method == "bilateral":
        return cv2.bilateralFilter(image, d=ks, sigmaColor=75, sigmaSpace=75)
    elif method == "nlm":
        return cv2.fastNlMeansDenoisingColored(
            image, None, h, h, template_window, search_window
        )
    else:
        raise ValueError(f"Unknown denoise method: '{method}'. "
                         "Choose from: gaussian, median, bilateral, nlm")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Contrast enhancement  (useful for dark space imagery)
# ─────────────────────────────────────────────────────────────────────────────
def enhance_contrast(
    image: Image,
    method: str = "clahe",
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
    alpha: float = 1.3,
    beta: float = 10,
) -> Image:
    """
    Enhance image contrast.

    Args:
        image:          uint8 BGR image.
        method:         'clahe' | 'linear'.
        clip_limit:     CLAHE clip limit (contrast limiting threshold).
        tile_grid_size: CLAHE tile grid size.
        alpha:          Linear contrast multiplier (method='linear').
        beta:           Linear brightness offset (method='linear').

    Returns:
        Contrast-enhanced uint8 BGR image.
    """
    if method == "clahe":
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        l_channel = clahe.apply(l_channel)
        enhanced_lab = cv2.merge([l_channel, a_channel, b_channel])
        return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    elif method == "linear":
        return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    else:
        raise ValueError(f"Unknown contrast method: '{method}'. "
                         "Choose from: clahe, linear")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Full preprocessing pipeline
# ─────────────────────────────────────────────────────────────────────────────
def preprocess_for_inference(
    image: Image,
    target_size: int = 640,
    denoise: bool = False,
    denoise_method: str = "gaussian",
    enhance: bool = False,
    enhance_method: str = "clahe",
) -> Image:
    """
    End-to-end preprocessing for a single inference image.

    Pipeline:
        [optional denoise] → [optional contrast enhance] → resize / letterbox

    Args:
        image:           uint8 BGR image (e.g. from cv2.imread).
        target_size:     Model input size.
        denoise:         Apply denoising step.
        denoise_method:  Denoising algorithm to use.
        enhance:         Apply contrast enhancement.
        enhance_method:  Contrast algorithm to use.

    Returns:
        Preprocessed uint8 BGR image ready for YOLO inference.
    """
    if image is None or image.size == 0:
        raise ValueError("preprocess_for_inference received an empty image.")

    if denoise:
        image = denoise_image(image, method=denoise_method)

    if enhance:
        image = enhance_contrast(image, method=enhance_method)

    image = resize_image(image, target_size=target_size, keep_aspect=True)
    return image


# ─────────────────────────────────────────────────────────────────────────────
# 6. Batch preprocessing for dataset preparation
# ─────────────────────────────────────────────────────────────────────────────
def preprocess_dataset_split(
    input_dir: Path,
    output_dir: Path,
    target_size: int = 640,
    denoise: bool = False,
    enhance: bool = True,
    exts: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tiff"),
) -> int:
    """
    Resize + optionally enhance all images in *input_dir* and write to
    *output_dir*, preserving filenames.

    Args:
        input_dir:   Source directory containing raw images.
        output_dir:  Destination directory (created if absent).
        target_size: Output image size (longest edge).
        denoise:     Apply Gaussian denoising.
        enhance:     Apply CLAHE contrast enhancement.
        exts:        Accepted image file extensions.

    Returns:
        Number of images processed.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    images = [p for p in input_dir.iterdir() if p.suffix.lower() in exts]

    if not images:
        print(f"[WARN] No images found in {input_dir}")
        return 0

    processed = 0
    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARN] Skipping unreadable file: {img_path.name}")
            continue
        img = preprocess_for_inference(
            img,
            target_size=target_size,
            denoise=denoise,
            enhance=enhance,
        )
        out_path = output_dir / img_path.name
        cv2.imwrite(str(out_path), img)
        processed += 1

    print(f"[INFO] Preprocessed {processed}/{len(images)} images → {output_dir}")
    return processed


# ─────────────────────────────────────────────────────────────────────────────
# 7. Utility: load image safely
# ─────────────────────────────────────────────────────────────────────────────
def load_image(path: str | Path) -> Image:
    """
    Read an image from disk.  Raises FileNotFoundError / ValueError on failure.

    Args:
        path: File path string or Path object.

    Returns:
        uint8 BGR numpy array.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"OpenCV could not read image: {path}")
    return img


# ─────────────────────────────────────────────────────────────────────────────
# Quick smoke test when run directly
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python src/preprocess.py <image_path> [--show]")
        sys.exit(0)

    src_path = sys.argv[1]
    show = "--show" in sys.argv

    img = load_image(src_path)
    print(f"Original : {img.shape}")

    out = preprocess_for_inference(img, denoise=True, enhance=True)
    print(f"Processed: {out.shape}")

    if show:
        cv2.imshow("Preprocessed", out)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
