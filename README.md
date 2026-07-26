# Air Quality Index (AQI) Prediction System

A web app that predicts AQI from pollutant and weather readings using both a
classical ML regression model and an ANN (TensorFlow/Keras), then shows the
AQI category and a health recommendation.

## Project Structure

```
AQI_Project/
├── dataset/
│   ├── generate_dataset.py   # creates a synthetic air_quality.csv (swap in real data later)
│   └── air_quality.csv
├── models/
│   ├── aqi_model.pkl         # best regression model (auto-selected)
│   ├── aqi_scaler.pkl
│   ├── model_metrics.pkl     # comparison metrics for all trained models
│   ├── aqi_ann.h5            # ANN model (after running ann_model.py)
│   └── ann_scaler.pkl
├── static/css/style.css
├── templates/                # index, predict, result, about
├── utils.py                  # shared feature list + AQI category logic
├── train_model.py            # trains & compares 5-6 regression models
├── ann_model.py               # trains the ANN
├── app.py                    # Flask app
└── requirements.txt
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 1. Get a dataset

A synthetic dataset is already generated at `dataset/air_quality.csv` so you
can run the whole pipeline immediately. To use real data, replace that file
with a real dataset (e.g. an India Air Quality / UCI AQI dataset) that has
these exact column names:

```
PM2.5, PM10, NO2, SO2, CO, O3, NH3, Temperature, Humidity, WindSpeed, AQI
```

(Or re-run `python dataset/generate_dataset.py` to regenerate the synthetic one.)

## 2. Train the models

```bash
python train_model.py     # trains & compares Linear/Tree/RF/GB/SVR/XGBoost, saves the best
python ann_model.py       # trains the TensorFlow ANN (requires tensorflow installed)
```

Both scripts read `dataset/air_quality.csv` and write to `models/`. The Flask
app works with just `train_model.py`'s output — the ANN is optional; if
`models/aqi_ann.h5` doesn't exist, the "ANN" option on the predict page is
simply disabled.

## 3. Run the app

```bash
python app.py
```

Visit `http://127.0.0.1:5000`.

## Notes on the included synthetic dataset

`dataset/generate_dataset.py` builds a realistic but **synthetic** dataset
(8,000 rows, with some missing values and duplicates injected on purpose) so
every part of the pipeline — cleaning, EDA, training, saving — has real work
to do out of the box. It is **not real air-quality data**; for a genuine
project, swap in an actual dataset with matching column names before your
final training run.

## Deployment

The app is a standard Flask app (`gunicorn app:app` works for Render/Railway).
Make sure `models/aqi_model.pkl` and `models/aqi_scaler.pkl` (and optionally
`models/aqi_ann.h5` / `models/ann_scaler.pkl`) are trained and present before
deploying — train them locally and commit them, or run the training scripts
as part of your build step.
"# PredictAirQualityIndex" 
