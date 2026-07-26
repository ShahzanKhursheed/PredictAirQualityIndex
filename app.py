"""
app.py
------
Flask backend for the AQI Prediction web app.

Routes:
    /            Home page
    /predict     GET: show the input form. POST: run prediction, show result.
    /about       Project information page

Run:
    python app.py
Then open http://127.0.0.1:5000
"""

import os

import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request

from utils import FEATURE_COLUMNS, get_aqi_category
from visualization_data import build_visualization_dashboard

app = Flask(__name__)

MODEL_PATH = "models/aqi_model.pkl"
SCALER_PATH = "models/aqi_scaler.pkl"
ANN_MODEL_PATH = "models/aqi_ann.h5"
ANN_SCALER_PATH = "models/ann_scaler.pkl"

# --- Load the regression model + scaler (required) ---------------------
ml_model = None
ml_scaler = None
if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
    ml_model = joblib.load(MODEL_PATH)
    ml_scaler = joblib.load(SCALER_PATH)

# --- Load the ANN model + scaler (optional, only if trained) ------------
ann_model = None
ann_scaler = None
if os.path.exists(ANN_MODEL_PATH) and os.path.exists(ANN_SCALER_PATH):
    try:
        import tensorflow as tf

        ann_model = tf.keras.models.load_model(ANN_MODEL_PATH, compile=False)
        ann_scaler = joblib.load(ANN_SCALER_PATH)
    except ImportError:
        # TensorFlow not installed -- ANN option simply won't be available.
        ann_model = None


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/visualizations")
def visualizations():
    dashboard = build_visualization_dashboard(request.args.get("city"))
    return render_template("visualizations.html", dashboard=dashboard)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return render_template(
            "predict.html", ann_available=ann_model is not None
        )

    # --- POST: read form inputs ---
    try:
        input_values = [float(request.form[col]) for col in FEATURE_COLUMNS]
    except (KeyError, ValueError):
        return render_template(
            "predict.html",
            ann_available=ann_model is not None,
            error="Please fill in all fields with valid numbers.",
        )

    model_choice = request.form.get("model_choice", "ml")
    X = pd.DataFrame([input_values], columns=FEATURE_COLUMNS)

    if model_choice == "ann" and ann_model is not None:
        X_scaled = ann_scaler.transform(X)
        prediction = float(ann_model.predict(X_scaled, verbose=0).flatten()[0])
        model_used = "Artificial Neural Network (ANN)"
    else:
        if ml_model is None:
            return render_template(
                "predict.html",
                ann_available=ann_model is not None,
                error="No trained model found. Run train_model.py first.",
            )
        X_scaled = ml_scaler.transform(X)
        prediction = float(ml_model.predict(X_scaled)[0])
        model_used = "Machine Learning Regression"

    prediction = max(0, round(prediction))
    category, advice = get_aqi_category(prediction)

    return render_template(
        "result.html",
        aqi=prediction,
        category=category,
        advice=advice,
        model_used=model_used,
        inputs=dict(zip(FEATURE_COLUMNS, input_values)),
    )


if __name__ == "__main__":
    app.run(debug=True)
