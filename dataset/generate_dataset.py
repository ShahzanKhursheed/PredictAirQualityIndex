"""
generate_dataset.py
--------------------
Generates a synthetic but realistic Indian air-quality dataset so the rest
of the pipeline (EDA, training, ANN, Flask app) can run end-to-end out of
the box.

IMPORTANT: This is a STAND-IN dataset. For a real project, replace
dataset/air_quality.csv with a real dataset (e.g. the "India Air Quality
Data" or a UCI AQI dataset) that has the SAME column names used below:

    PM2.5, PM10, NO2, SO2, CO, O3, NH3, Temperature, Humidity, WindSpeed, AQI

As long as the column names match, train_model.py and ann_model.py will
work unchanged on the real data.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N = 8000


def generate_dataset(n_rows: int = N) -> pd.DataFrame:
    # Base pollutant concentrations (log-normal-ish, realistic ranges for Indian cities)
    pm25 = np.random.gamma(shape=2.2, scale=35, size=n_rows).clip(2, 500)
    pm10 = (pm25 * np.random.uniform(1.3, 2.2, n_rows)).clip(5, 600)
    no2 = np.random.gamma(shape=2.0, scale=15, size=n_rows).clip(2, 200)
    so2 = np.random.gamma(shape=1.8, scale=8, size=n_rows).clip(1, 120)
    co = np.random.gamma(shape=2.0, scale=0.6, size=n_rows).clip(0.1, 10)
    o3 = np.random.gamma(shape=2.2, scale=20, size=n_rows).clip(2, 200)
    nh3 = np.random.gamma(shape=2.0, scale=10, size=n_rows).clip(1, 150)

    temperature = np.random.normal(27, 7, n_rows).clip(2, 48)
    humidity = np.random.normal(55, 20, n_rows).clip(5, 100)
    wind_speed = np.random.gamma(shape=2.0, scale=1.5, size=n_rows).clip(0.1, 20)

    # AQI approximated as a weighted, noisy, non-linear combination of pollutants.
    # This mimics how PM2.5/PM10 dominate the Indian AQI sub-index in practice.
    aqi = (
        1.6 * pm25
        + 0.55 * pm10
        + 0.9 * no2
        + 0.6 * so2
        + 12 * co
        + 0.35 * o3
        + 0.25 * nh3
        - 0.3 * wind_speed
        + np.random.normal(0, 18, n_rows)
    )
    aqi = aqi.clip(5, 500)

    df = pd.DataFrame(
        {
            "PM2.5": pm25.round(2),
            "PM10": pm10.round(2),
            "NO2": no2.round(2),
            "SO2": so2.round(2),
            "CO": co.round(2),
            "O3": o3.round(2),
            "NH3": nh3.round(2),
            "Temperature": temperature.round(2),
            "Humidity": humidity.round(2),
            "WindSpeed": wind_speed.round(2),
            "AQI": aqi.round(0).astype(int),
        }
    )

    # Inject a few missing values and duplicates so the preprocessing code
    # in train_model.py has something real to clean.
    for col in ["PM2.5", "Humidity", "O3"]:
        idx = np.random.choice(df.index, size=int(0.01 * n_rows), replace=False)
        df.loc[idx, col] = np.nan

    dup_idx = np.random.choice(df.index, size=int(0.005 * n_rows), replace=False)
    df = pd.concat([df, df.loc[dup_idx]], ignore_index=True)

    return df


if __name__ == "__main__":
    data = generate_dataset()
    out_path = "dataset/air_quality.csv"
    data.to_csv(out_path, index=False)
    print(f"Saved {len(data)} rows to {out_path}")
    print(data.head())
