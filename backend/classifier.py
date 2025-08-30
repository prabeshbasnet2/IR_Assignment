# classifier.py
import joblib
from pathlib import Path

MODEL_PATH = Path("models/news_clf.joblib")

_model = None

def load_model():
    global _model
    if _model is None:
        payload = joblib.load(MODEL_PATH)
        _model = payload["pipeline"]
    return _model

def classify_text(text: str):
    model = load_model()
    pred = model.predict([text])[0]
    probs = None
    try:
        probs = model.predict_proba([text])[0].max()
    except Exception:
        pass
    return {"label": pred, "confidence": float(probs) if probs is not None else None}
