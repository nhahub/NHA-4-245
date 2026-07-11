# Land Cover Classifier — Web UI

A simple Streamlit web app for your EfficientNet land-cover classification model.

## Setup

1. Put `EfficientNet_final.h5` in this same folder (next to `app.py`).
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the app:
   ```
   streamlit run app.py
   ```
4. Your browser will open automatically at `http://localhost:8501`.

## Using it

- Upload a `.tif` file with 13 bands (Sentinel-2 / EuroSAT style imagery).
- The app shows an RGB preview built from bands 4/3/2, plus basic image info.
- Click **Run prediction** to classify the image and see confidence scores
  for all 10 classes.

## Notes

- The model is loaded once and cached, so predictions after the first one
  are faster.
- If you get a "couldn't load model" error, double check the filename
  matches `EfficientNet_final.h5` exactly (or edit `MODEL_PATH` in `app.py`).
