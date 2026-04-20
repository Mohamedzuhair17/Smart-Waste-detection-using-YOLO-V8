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
