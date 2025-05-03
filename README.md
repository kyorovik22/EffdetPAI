# 🩻 X-Ray Object Detection Web App using  Flask

A web-based application for detecting medical conditions or anomalies in X-ray images using **EFFDET** for object detection and **Flask** as the backend web framework. This app includes preprocessing steps such as auto-orientation, histogram equalization, and resizing to ensure input consistency.

---

## 🚀 Features

- ✅ Upload X-ray images via browser
- ✅ Automatic preprocessing:
  - EXIF auto-orientation
  - Histogram equalization (contrast enhancement)
  - Resize to 640×640 (stretch)
- ✅ EFFDET detection with bounding boxes and class labels
- ✅ Prediction results with confidence score
- ✅ Clean UI built with HTML and Flask templating

---

## 🧰 Tech Stack

- Python 3.8+
- Flask
- OpenCV
- HTML/CSS (Jinja2 templating)

---

## 📁 Directory Structure

├── app.py 
├── models/
│ └── best.pt 
├── uploads/ 
├── templates/ 
│ ├── index.html
│ ├── detection.html
│ ├── result.html
│ └── about.html
├── static/ 
├── requirements.txt 
└── .gitignore 


---

## 🛠️ Installation Guide

## 🛠️ Installation Guide

### 1. Clone this repository

```bash
git clone https://github.com/kyorovik22/EffdetPAI.git
cd EffdetPAI
```

### 2. Create and activate a virtual environment

#### On **Windows**:

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Flask app

```bash
python app.py
```

Visit the app at:

```
http://127.0.0.1:5000/
```

---
