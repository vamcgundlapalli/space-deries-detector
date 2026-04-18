# 🛸 AI-Powered Space Debris Detection System

> Real-time orbital debris detection using **YOLOv8**, **OpenCV**, **Streamlit**, **FastAPI**, and **HTML/JS Frontend**

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-black.svg)
![HTML/CSS/JS](https://img.shields.io/badge/HTML/CSS/JS-blueviolet.svg)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Dataset Setup](#dataset-setup)
- [Training](#training)
- [Inference](#inference)
- [Streamlit App](#streamlit-app)
- [HTML/JS Frontend](#htmljs-frontend)
- [Dataset Annotation Format](#dataset-annotation-format)
- [Expected Outputs](#expected-outputs)
- [Tech Stack](#tech-stack)

---

## Overview

This system uses a fine-tuned **YOLOv8** model to detect and classify three categories of orbital objects in still images and live video:

| Class | ID | Description |
|---|---|---|
| `debris` | 0 | Generic orbital debris fragments |
| `defunct_satellite` | 1 | Non-operational spacecraft |
| `rocket_body` | 2 | Spent rocket stages and boosters |

**Dual UIs:**
- **Streamlit** (Python): Full-featured dashboard (app/app.py)
- **HTML/JS Frontend** (Browser): Pure client-side with FastAPI backend (frontend/, api/app.py)

---

## Features

- **YOLOv8 object detection** — state-of-the-art single-shot detection
- **Image detection** — upload any image and get annotated results instantly
- **Real-time webcam detection** — live debris detection (~10 FPS in browser)
- **Adjustable confidence threshold** — tune precision vs. recall interactively
- **CLAHE contrast enhancement** — improved detection in dark space imagery
- **Gaussian denoising** — reduce sensor noise before inference
- **Per-class detection counts** — break down results by object category
- **Annotated image download** — save results as PNG
- **Synthetic dataset generator** — get started without any real data
- **Dark space-themed UI** — both Streamlit & HTML match design
- **FastAPI backend** — REST API for frontend integration

---

## Project Structure

```
space-debris-detection/
│
├── dataset/           # YOLO dataset (images/labels)
├── models/            # Trained YOLOv8 weights (*.pt)
├── src/               # Core Python: train.py, detect.py, preprocess.py
├── app/               # Streamlit UI (Python)
│   └── app.py
├── api/               # FastAPI REST API backend
│   └── app.py
├── frontend/          # HTML/CSS/JS UI (browser-native)
│   ├── index.html
│   ├── style.css
│   └── script.js
├── scripts/           # Dataset helpers
├── runs/              # Auto-generated outputs
├── requirements.txt   # pip deps (ultralytics, torch, streamlit, fastapi...)
└── README.md
```

---

## Installation

```bash
# 1. Activate venv
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate

# 2. Install all dependencies (Streamlit + FastAPI + Frontend)
pip install -r requirements.txt
```

**GPU**: Use CUDA torch from https://pytorch.org/get-started/locally/

---

## Dataset Setup

(Same as original - synthetic/Kaggle/custom via `scripts/download_dataset.py`)

---

## Training

(Same CLI as original - `python src/train.py ...`)

---

## Inference

**CLI** (same): `python src/detect.py --source img.jpg --weights best.pt --conf 0.25`

---

## Streamlit App

**Original Python UI:**
```bash
streamlit run app/app.py
```
Open http://localhost:8501

## HTML/JS Frontend

**Modern browser UI with API backend:**

1. **Start API server:**
   ```bash
   python app.py
   ```
   (Or: `uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload`)

2. **Open Frontend:**
   - Double-click `frontend/index.html` or
   - VSCode Live Server extension (`http://localhost:5500/frontend/index.html`)
   
   **API URL:** http://localhost:8000

**Features:**
- **📸 Image**: Upload → detect → orig/detected side-by-side + metrics + table + PNG download
- **📷 Webcam**: Live 10FPS detection with box overlays + FPS counter
- **Sidebar**: Live sliders (conf/iou/imgsz), checkboxes (denoise/enhance)
- **Responsive**: Mobile-friendly dark space theme

**API Endpoint:** `POST http://localhost:8000/detect`
- Multipart image + query params → JSON + annotated PNG b64

---

## Dataset Annotation Format

(Same YOLO txt format)

---

## Expected Outputs

**API Response Example:**
```json
{
  "success": true,
  "detections": [{"class_name": "debris", "confidence": 0.891, "bbox": [142,87,298,201]}],
  "counts": {"debris": 2, "defunct_satellite": 1},
  "total_detections": 3,
  "inference_ms": 14.2,
  "annotated_image_b64": "data:image/png;base64,iVBORw0K...",
  "image_shape": [480, 640, 3]
}
```

---

## Tech Stack

| Component | Backend | Frontend |
|-----------|---------|----------|
| **Detection** | YOLOv8 | - |
| **DL Framework** | PyTorch | - |
| **Vision** | OpenCV | Canvas 2D |
| **API** | FastAPI | Fetch API |
| **UIs** | Streamlit \| HTML/JS | - |
| **Language** | Python 3.10+ | - |

---

## Contributing / License / References

(Same as original)

**New:** Pure HTML/JS frontend works offline (except API calls) - no Node/build tools needed!

