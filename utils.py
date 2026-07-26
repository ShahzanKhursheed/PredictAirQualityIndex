"""
utils.py
--------
Shared helpers used by train_model.py, ann_model.py, and app.py.
Keeping this logic in one place avoids the category thresholds / feature
list drifting out of sync between training and serving.
"""

FEATURE_COLUMNS = [
    "PM2.5",
    "PM10",
    "NO2",
    "SO2",
    "CO",
    "O3",
    "NH3",
    "Temperature",
    "Humidity",
    "WindSpeed",
]

TARGET_COLUMN = "AQI"

# (upper_bound, category, health_advice)
AQI_CATEGORIES = [
    (50, "Good", "Air quality is satisfactory. Enjoy your usual outdoor activities."),
    (100, "Satisfactory", "Minor breathing discomfort possible for sensitive people."),
    (200, "Moderate", "Sensitive people (asthma, heart/lung conditions, children, elderly) should reduce prolonged outdoor exertion."),
    (300, "Poor", "Everyone may begin to experience breathing discomfort on prolonged exposure. Limit outdoor activity."),
    (400, "Very Poor", "Health warnings of emergency conditions. Avoid outdoor activity, especially for sensitive groups."),
    (500, "Severe", "Serious health effects for everyone. Stay indoors and use an air purifier if possible."),
]


def get_aqi_category(aqi_value: float):
    """Return (category, health_advice) for a given AQI value."""
    aqi_value = max(0, min(500, aqi_value))
    for upper_bound, category, advice in AQI_CATEGORIES:
        if aqi_value <= upper_bound:
            return category, advice
    return AQI_CATEGORIES[-1][1], AQI_CATEGORIES[-1][2]


def clean_dataframe(df):
    """Basic cleaning shared by training scripts: dedupe + drop missing target rows."""
    df = df.drop_duplicates().copy()
    # Impute missing feature values with column median (robust to outliers)
    for col in FEATURE_COLUMNS:
        if col in df.columns and df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    df = df.dropna(subset=[TARGET_COLUMN])
    return df
