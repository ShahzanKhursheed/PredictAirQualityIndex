"""
ann_model.py
------------
Trains an Artificial Neural Network (TensorFlow/Keras) to predict AQI and
saves it to models/aqi_ann.h5, plus its own scaler to models/ann_scaler.pkl.

Run:
    python ann_model.py

Requires: tensorflow (pip install tensorflow)
"""

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, models

from utils import FEATURE_COLUMNS, TARGET_COLUMN, clean_dataframe

DATA_PATH = "dataset/air_quality.csv"
ANN_MODEL_PATH = "models/aqi_ann.h5"
ANN_SCALER_PATH = "models/ann_scaler.pkl"

tf.random.set_seed(42)
np.random.seed(42)


def build_ann(input_dim: int) -> tf.keras.Model:
    model = models.Sequential(
        [
            layers.Input(shape=(input_dim,)),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(32, activation="relu"),
            layers.Dense(1, activation="linear"),  # regression output
        ]
    )
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def main():
    print("Loading and preparing data...")
    df = pd.read_csv(DATA_PATH)
    df = clean_dataframe(df)

    X = df[FEATURE_COLUMNS].values
    y = df[TARGET_COLUMN].values.astype("float32")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = build_ann(input_dim=X_train_scaled.shape[1])
    model.summary()

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=15, restore_best_weights=True
    )

    print("\nTraining ANN...")
    history = model.fit(
        X_train_scaled,
        y_train,
        validation_split=0.15,
        epochs=150,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1,
    )

    preds = model.predict(X_test_scaled).flatten()
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"\nANN Test MAE: {mae:.2f}  R2: {r2:.4f}")

    model.save(ANN_MODEL_PATH)
    joblib.dump(scaler, ANN_SCALER_PATH)
    print(f"\nSaved ANN model to {ANN_MODEL_PATH}")
    print(f"Saved ANN scaler to {ANN_SCALER_PATH}")


if __name__ == "__main__":
    main()
