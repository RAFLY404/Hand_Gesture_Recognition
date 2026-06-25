# Hand Gesture Recognition Streamlit App

This project is packaged for Streamlit deployment. The app loads the trained gesture model, scaler, and label encoder from the existing pickle files, applies the same preprocessing and feature extraction used in `hgr.ipynb`, and predicts one of five gestures from a live browser webcam stream: fist, open hand, peace, pointing, or thumbs up.

## Project Files

- `app.py` - Streamlit WebRTC live detection app.
- `gesture_predictor.py` - reusable preprocessing, feature extraction, and prediction logic.
- `gesture_model.pkl` - trained Random Forest classifier.
- `gesture_scaler.pkl` - fitted StandardScaler.
- `gesture_encoder.pkl` - fitted LabelEncoder.
- `requirements.txt` - Python dependencies for Streamlit Cloud.
- `.streamlit/config.toml` - Streamlit runtime and theme config.

## Run Locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

After the app opens, click **Start**, allow browser camera access, and place your hand inside the green box.
