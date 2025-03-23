# WebXR 3D Image Inference Viewer

This project uses a Flask backend to run object detection, segmentation, and monocular depth estimation. The frontend includes a WebXR stereo viewer built with Three.js for immersive stereo image exploration.

---

## Environment Setup (Conda)

Follow the steps below to create a clean conda environment and install all required dependencies.

### 1. Create a new conda environment with Python 3.8.18

```bash
conda create -n webxr3d-env python=3.8.18
```

### 2. Activate the environment

```bash
conda activate webxr3d-env
```

### 3. Install core libraries using `conda-forge`

This will install Flask for the web app and the necessary libraries for image processing and deep learning.

```bash
conda install flask opencv matplotlib numpy pytorch torchvision -c conda-forge
```

### 4. Install `timm` (PyTorch Image Models) via pip

```bash
pip install timm
```

---

## Run the Flask App

Once all dependencies are installed, you can start the web server:

```bash
python app.py
```

By default, the app will be available at:  
📍 `http://127.0.0.1:5000/`

---

## Project Structure

```bash
.
├── app.py                    # Main Flask server
├── inference_pipeline.py     # Core inference logic
├── yolov8s.pt                # YOLOv8 weights
├── uploads/                  # Uploaded input images
├── outputs/                  # Processed image results
├── templates/
│   ├── index.html            # Upload form and image display UI
│   └── stereo.html           # WebXR stereo viewer page
└── README.md                 # Setup instructions (this file)
```

---

## WebXR Support

To use the WebXR stereo viewer:
- Use a compatible browser like **Chrome**, **Edge**, or **Oculus Browser**
- If you're testing VR on a headset (e.g. Meta Quest), open `http://<your-ip>:5000/stereo-view/<image-name>`
- For desktop testing, ensure `chrome://flags/#webxr` is enabled

---

## Notes

- Python 3.8 is recommended for compatibility with PyTorch and legacy image libs.
- `timm` is required if you're extending the model or importing certain vision backbones.

---