# NHA-4-245
Auto generated repo 245

# Satellite Land Cover Classifier

A deep-learning pipeline that trains and compares four CNN architectures — **SpectrumNet**, **SimpleCNN**, **ResNet50**, and **EfficientNet** — on 14-channel EuroSAT satellite imagery (13 Sentinel-2 bands + a computed NDVI channel), classifying scenes into 10 land-cover classes. A companion **Streamlit** app serves the trained models for interactive inference.

| | |
|---|---|
| **Training pipeline** | `Final_EuroSat.ipynb` (Jupyter notebook) |
| **Inference app** | `app.py` (Streamlit) |
| **Dataset** | EuroSAT, 13-band Sentinel-2 imagery, 27,597 images, 10 classes |
| **Best result** | SpectrumNet — 97.4% test accuracy |

---

## 1. System Requirements

### Hardware

| Component | Minimum | Recommended |
|---|---|---|
| RAM | 8 GB | 16 GB+ (training uses large in-memory arrays) |
| GPU | Not required for the app | NVIDIA GPU with CUDA support, for training only |
| Disk | ~2 GB free | 5 GB+ (dataset + 4 saved `.h5` models; ResNet50 alone is ~277 MB) |

### Software

| Dependency | Version used | Needed for |
|---|---|---|
| Python | 3.10+ | Both |
| TensorFlow / Keras | 2.x | Both |
| Streamlit | 1.3x+ | App only |
| NumPy, Pandas | latest | Both |
| scikit-learn | latest | Notebook only (splits, metrics, PCA) |
| scikit-image | latest | Both (`resize`) |
| tifffile | latest | Both (`.tif` I/O) |
| seaborn, matplotlib | latest | Notebook only (plots, confusion matrices, Grad-CAM) |
| opencv-python (`cv2`) | latest | Notebook only (Grad-CAM overlay) |

A GPU is **not** required to run the Streamlit app — inference is fast on CPU. It is strongly recommended for re-running the training notebook.

---

## Repository Structure

The repository is organized into two main directories:

### `documentation/`

Contains all project documentation and supporting materials, including:

- System Analysis & Design documentation
- Requirements Gathering document
- Other project-related reports and documents

### `finally/`

Contains the final implementation and generated outputs of the project, including:

- `Final_EuroSat.ipynb` – Complete training and evaluation notebook.
- `Model_Comparison.csv` – Comparison of all trained models and their evaluation metrics.
- `PCA_2D_Visualization.png` – Two-dimensional PCA visualization of the processed dataset.
- `PCA_ExplainedVariance.png` – Explained variance plot showing the cumulative variance retained by PCA.

These files represent the final results produced during the implementation and evaluation stages of the project.

---



## 2. Installation

```bash
# 1. Clone or copy the project files into a working directory
cd satellite-land-cover-classifier

# 2. Create and activate a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install streamlit tensorflow numpy pandas scikit-learn scikit-image tifffile
# Notebook-only extras (only needed to re-run training):
pip install matplotlib seaborn opencv-python jupyter
```

> **Note:** No `requirements.txt` is bundled with this project yet. If you add one, pin the versions above so the app and notebook stay reproducible.

---

## 3. Configuration


### 3.1 Model files

`app.py` loads four pre-trained model files, referenced in the `MODEL_REGISTRY` dict at the top of the file:

```python
MODEL_REGISTRY = {
    "EfficientNet": {"file": "EfficientNet_final.h5", ...},
    "ResNet50":     {"file": "ResNet50_final.h5", ...},
    "SimpleCNN":    {"file": "SimpleCNN_final.h5", ...},
    "SpectrumNet":  {"file": "SpectrumNet_final.h5", ...},
}
```

Place all four `.h5` files **in the same directory as `app.py`** (paths are relative, no hard-coded absolute paths — see `NFR-8` in the requirements docs). These files are produced by running the training notebook, or can be copied from wherever training was previously run.

### 3.2 Dataset path (training only)

The notebook points at a local dataset folder:

```python
DATA_ROOT = r"C:\college\CourseDataSci\final 2026\EuroSAT\allBands"
```

Update `DATA_ROOT` to wherever your EuroSAT 13-band `allBands` dataset lives before re-running training. The loader expects one subfolder per class, each containing 13-band `.tif` files.

### 3.3 Reproducibility

Training uses a fixed random seed (`42`) for the stratified 80/10/10 train/validation/test split, so re-running training on the same data reproduces the same split.

---

## 4. Execution Guide

### 4.1 Run the app locally

```bash
streamlit run app.py
```

This opens the app at `http://localhost:8501`. From there:

1. Pick a model (transfer-learning or lightweight card).
2. Upload a genuine **13-band `.tif`** satellite image.
3. Review the true-color RGB preview.
4. Click **Run prediction** to see the predicted class, confidence, and full 10-class probability breakdown.

The app validates that an upload has exactly 13 bands, resizes it to 64×64, computes an NDVI channel from the Red/NIR bands, and feeds the resulting 14-channel tensor to the selected model — the same preprocessing used in training.

### 4.2 Re-run training (optional)

```bash
jupyter notebook Final_EuroSat__Edited_.ipynb
```

Run cells top to bottom. Key stages: load & preprocess data → compute NDVI → encode labels → split 80/10/10 → train each of the 4 architectures (up to 40 epochs, early stopping on `val_loss`, patience 7) → evaluate → compare → (optional) confusion matrices, ROC curves, PCA, Grad-CAM. Each model is checkpointed to `<ModelName>_best.h5` and saved as `<ModelName>_final.h5`.

### 4.3 Accessing a deployed version

No hosted/deployed instance currently exists for this project. See [Section 6](#6-executable-files--deployment) for how to deploy one.

---

## 5. API Documentation

There is no REST API in this project today — `app.py` is a single Streamlit process with no HTTP layer. The table below documents the internal function contract a future API could wrap directly.

| Function | Input | Output | Purpose |
|---|---|---|---|
| `preprocess_image(uploaded_file)` | Uploaded `.tif` file | `(raw_img: ndarray, model_input: ndarray)` | Validates 13 bands, resizes to 64×64, computes NDVI, returns a 14-channel model input |
| `add_ndvi_channel(img)` | 13 (or 13-band-resized) ndarray | 14-channel ndarray | Appends `(NIR − Red) / (NIR + Red)` as a new channel |
| `make_rgb_preview(img)` | 13-band ndarray | RGB ndarray (0–1 range) | Builds a true-color preview from bands 4/3/2 |
| `get_model(model_path)` | Model file path (str) | Loaded Keras model (cached via `st.cache_resource`) | Loads a `.h5` model once per path, reused on subsequent calls |
| `model.predict(model_input)` | Preprocessed `(1, 64, 64, 14)` array | `float[10]` class probabilities | Runs inference for the selected model |

**Future extensibility:** a `POST /predict` endpoint could wrap `preprocess_image()` + `model.predict()` directly, accepting a `.tif` upload and a model name, and returning the same class/confidence/probability JSON shape shown in the app's results panel.

---

## 6. Executable Files & Deployment

### 6.1 Compiled / packaged application

This project is a Python + Streamlit application, not a compiled desktop or mobile app — there is no `.exe`, `.jar`, or `.apk` build, and none is planned, since the deliverable is a web-based inference tool.

If a standalone package is needed:
- **Desktop-style wrapper:** package with [PyInstaller](https://pyinstaller.org/) or run inside a lightweight Electron shell that launches `streamlit run app.py` and opens a browser view.
- **Containerized package:** build a Docker image (see below) — this is the recommended path for anything beyond local use.

### 6.2 Deployed web app

**No live deployment currently exists.** To deploy one:

**Option A — Streamlit Community Cloud** (fastest for a demo):
1. Push `app.py` + the four `.h5` model files + a `requirements.txt` to a GitHub repo.
2. Connect the repo at [share.streamlit.io](https://share.streamlit.io) and deploy.
3. Note: Streamlit Community Cloud has repo size limits — the 277 MB ResNet50 file may need Git LFS or external storage.

**Option B — Docker container on your own infrastructure:**
```bash
# Example Dockerfile outline
FROM python:3.10-slim
WORKDIR /app
COPY . .
RUN pip install streamlit tensorflow numpy pandas scikit-learn scikit-image tifffile
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
```
```bash
docker build -t land-cover-classifier .
docker run -p 8501:8501 land-cover-classifier
```

Once deployed, replace this section with the live URL and update Section 4.3 accordingly.

### 6.3 Deployment architecture (current design)

A single Streamlit process serves the browser directly and reads `.h5` model files from local disk/mounted volume — there is no database or REST tier yet:

```
Client Browser  --HTTPS-->  Streamlit App (single process)  --reads-->  Model files (4 × .h5, local disk)
```

Model caching is per-process, so a multi-instance deployment reloads models independently per instance. The heaviest model (ResNet50, ~277 MB) should be lazy-loaded only when selected to control memory.

---

## 7. Project Structure

```
.
├── app.py                              # Streamlit inference app
├── Final_EuroSat__Edited_.ipynb        # Training notebook (data → 4 models → evaluation)
├── SpectrumNet_final.h5                # Trained model (produced by the notebook)
├── SimpleCNN_final.h5                  # Trained model (produced by the notebook)
├── ResNet50_final.h5                   # Trained model (produced by the notebook)
├── EfficientNet_final.h5               # Trained model (produced by the notebook)
├── System_Analysis_Design.pdf          # System design documentation
├── Requirements_Gathering.pdf          # Requirements documentation
├── Project_Presentation.pptx / .pdf    # Project summary presentation
└── README.md                           # This file
```

---

## 8. Model Results (from the training notebook)

Weighted-average metrics on the held-out 2,760-image test set:

| Model | Type | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| **SpectrumNet** | Lightweight custom CNN | **97.4%** | 97.4% | 97.4% | 97.4% |
| EfficientNet | Transfer learning | 96.2% | 96.2% | 96.2% | 96.2% |
| ResNet50 | Transfer learning | 94.7% | 94.8% | 94.7% | 94.7% |
| SimpleCNN | Custom CNN | 93.7% | 93.8% | 93.7% | 93.7% |

The lightweight, purpose-built SpectrumNet (~1 MB) outperformed both transfer-learning backbones despite being a fraction of their size.

---

## 9. Known Limitations

- No automated test suite is included yet (see `System_Analysis_Design.pdf` → Testing & Validation for the planned unit/integration/UAT coverage).
- No REST API, authentication, or persistence layer — the app is stateless.
- No `requirements.txt` / dependency lock file is included; pin versions before deploying.
