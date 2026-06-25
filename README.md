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

## Deploy To Streamlit Community Cloud

1. Push this project to a GitHub repository.
2. Make sure `app.py`, `gesture_predictor.py`, `requirements.txt`, `gesture_model.pkl`, `gesture_scaler.pkl`, and `gesture_encoder.pkl` are included.
3. Go to Streamlit Community Cloud and create a new app from the GitHub repository.
4. Set the main file path to `app.py`.
5. Deploy the app.

The `gesture_dataset` folder is not required for deployment because the model has already been trained and saved.

Live webcam access requires HTTPS in production. Streamlit Community Cloud provides HTTPS automatically.
