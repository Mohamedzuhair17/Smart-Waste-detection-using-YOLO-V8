<!-- EFFECTS-BLOCK:START -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=rect&color=F6F1EA&height=180&section=header&text=Smart%20Waste%20Detection%20Using%20YOLO%20V8&fontSize=44&fontColor=111111&desc=Real-time%20waste%20classification%20inference%20engine%20using%20custom%20YOLOv8%20pipeline%20for%20production%20segregation.&descSize=14&descAlignY=68" alt="Smart Waste Detection Using YOLO V8" />
</p>

<p align="center">
  <a href="https://github.com/Mohamedzuhair17/Smart-Waste-detection-using-YOLO-V8"><img src="https://img.shields.io/badge/Repository-111111?style=for-the-badge&logo=github" alt="repo" /></a>
  <img src="https://img.shields.io/github/stars/Mohamedzuhair17/=for-the-badge&color=111111" alt="stars" />
  <img src="https://img.shields.io/github/forks/Mohamedzuhair17/=for-the-badge&color=111111" alt="forks" />
  <img src="https://img.shields.io/github/last-commit/Mohamedzuhair17/=for-the-badge&color=111111" alt="last commit" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Stack-Python-F6F1EA?style=for-the-badge&labelColor=111111&color=F6F1EA" alt="stack" />
  <img src="https://img.shields.io/badge/Engineering-Production%20Grade-111111?style=for-the-badge" alt="engineering" />
</p>
<!-- EFFECTS-BLOCK:END -->

---

# Smart Waste Detection using YOLOv8

AI-powered Streamlit app for detecting and classifying waste as **ORGANIC** or **NON-ORGANIC** using a trained YOLOv8 model.

## Demo

Add your demo assets in `assets/` and keep one of these in README:

```md
![App Screenshot](assets/app-screenshot.png)
```

```md
![Demo GIF](assets/demo.gif)
```

## Problem It Solves

Waste is often mixed at source, making recycling and disposal inefficient. This project helps with first-level segregation by classifying visible trash in uploaded images, enabling faster and smarter sorting decisions.

## Why This Project Matters

- Reduces manual effort in basic waste sorting workflows.
- Demonstrates practical AI for environmental use-cases.
- Provides an accessible UI (Streamlit) for non-technical users.

## Model Performance

### Detection Output
- Classes are normalized to two bins in app output: `ORGANIC` and `NON-ORGANIC`.

### Inference Benchmark (local CPU)
- Model file: `best.pt`
- Input shape used: `640 x 640` (synthetic image)
- Average inference time: **~391.96 ms / image**
- Fastest run: **121.36 ms**
- Slowest run: **3690.83 ms**

> Note: Throughput depends on hardware and image complexity. Add your dataset validation metrics (mAP/precision/recall) if available.

## Setup

### 1. Clone
```bash
git clone https://github.com/Mohamedzuhair17/Smart-Waste-detection-using-YOLO-V8.git
cd Smart-Waste-detection-using-YOLO-V8
```

### 2. Create Virtual Environment
```bash
python -m venv .venv
```

- Windows PowerShell:
```powershell
.\.venv\Scripts\Activate.ps1
```

- macOS/Linux:
```bash
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the App
```bash
streamlit run project.py
```

## Usage

1. Upload a trash image (`png/jpg/jpeg`).
2. Adjust confidence and IoU thresholds in sidebar.
3. Click **Detect Trash**.
4. Review bounding boxes and classification summary.

## Tech Stack

- Python
- Ultralytics YOLOv8
- Streamlit
- Pillow / NumPy
