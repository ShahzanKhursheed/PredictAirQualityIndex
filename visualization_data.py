from functools import lru_cache
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATASETS_DIR = BASE_DIR / "datasets"

CATEGORY_ORDER = [
    "Good",
    "Satisfactory",
    "Moderate",
    "Poor",
    "Very Poor",
    "Severe",
]

CATEGORY_COLORS = {
    "Good": "#2f9e44",
    "Satisfactory": "#74b816",
    "Moderate": "#f59f00",
    "Poor": "#f76707",
    "Very Poor": "#e03131",
    "Severe": "#862e9c",
    "Unknown": "#64748b",
}


def category_for_aqi(value):
    if value <= 50:
        return "Good"
    if value <= 100:
        return "Satisfactory"
    if value <= 200:
        return "Moderate"
    if value <= 300:
        return "Poor"
    if value <= 400:
        return "Very Poor"
    return "Severe"


def color_for_aqi(value):
    return CATEGORY_COLORS.get(category_for_aqi(value), CATEGORY_COLORS["Unknown"])


@lru_cache(maxsize=1)
def load_bulletin_data():
    columns = [
        "date",
        "City",
        "No. Stations",
        "Air Quality",
        "Index Value",
        "Prominent Pollutant",
    ]
    frames = []

    for path in sorted(DATASETS_DIR.glob("*_AQIBulletins.csv")):
        try:
            frame = pd.read_csv(
                path,
                usecols=lambda column: column in columns,
                encoding_errors="ignore",
            )
        except (OSError, pd.errors.ParserError, ValueError):
            continue

        if "date" not in frame.columns or "Index Value" not in frame.columns:
            continue

        fallback_city = path.stem.replace("_AQIBulletins", "").replace("_", " ")
        if "City" not in frame.columns:
            frame["City"] = fallback_city
        else:
            frame["City"] = frame["City"].fillna(fallback_city)

        if "Air Quality" not in frame.columns:
            frame["Air Quality"] = "Unknown"
        if "Prominent Pollutant" not in frame.columns:
            frame["Prominent Pollutant"] = "Unknown"
        if "No. Stations" not in frame.columns:
            frame["No. Stations"] = pd.NA

        frame["date"] = pd.to_datetime(
            frame["date"],
            errors="coerce",
            format="mixed",
            dayfirst=True,
        )
        frame["Index Value"] = pd.to_numeric(frame["Index Value"], errors="coerce")
        frame["No. Stations"] = pd.to_numeric(frame["No. Stations"], errors="coerce")
        frame["City"] = frame["City"].astype(str).str.strip()
        frame["Air Quality"] = frame["Air Quality"].astype(str).str.strip()
        frame["Prominent Pollutant"] = (
            frame["Prominent Pollutant"].astype(str).str.strip().str.upper()
        )

        frame = frame.dropna(subset=["date", "Index Value"])
        if not frame.empty:
            frames.append(frame[columns])

    if not frames:
        return pd.DataFrame(columns=columns)

    return pd.concat(frames, ignore_index=True)


@lru_cache(maxsize=1)
def load_station_data():
    path = DATASETS_DIR / "dataset1.csv"
    columns = [
        "state",
        "city",
        "station",
        "last_update",
        "latitude",
        "longitude",
        "pollutant_id",
        "pollutant_avg",
        "AQI",
        "AQI_Bucket",
        "Temperature_C",
        "Humidity_%",
        "Wind_Speed_kmh",
        "Season",
    ]

    if not path.exists():
        return pd.DataFrame(columns=columns)

    frame = pd.read_csv(
        path,
        usecols=lambda column: column in columns,
        encoding_errors="ignore",
    )

    for column in columns:
        if column not in frame.columns:
            frame[column] = pd.NA

    frame["last_update"] = pd.to_datetime(frame["last_update"], errors="coerce")
    numeric_columns = [
        "latitude",
        "longitude",
        "pollutant_avg",
        "AQI",
        "Temperature_C",
        "Humidity_%",
        "Wind_Speed_kmh",
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    text_columns = ["state", "city", "station", "pollutant_id", "AQI_Bucket", "Season"]
    for column in text_columns:
        frame[column] = frame[column].astype(str).str.strip()

    return frame


def build_visualization_dashboard(requested_city=None):
    bulletins = load_bulletin_data()
    stations = load_station_data()

    if bulletins.empty:
        return {"has_data": False, "cities": [], "selected_city": None}

    cities = sorted(bulletins["City"].dropna().unique().tolist())
    selected_city = _resolve_city(requested_city, cities)
    city_frame = bulletins[bulletins["City"] == selected_city].copy()
    city_frame = city_frame.sort_values("date")

    date_min = bulletins["date"].min()
    date_max = bulletins["date"].max()
    average_aqi = float(bulletins["Index Value"].mean())
    latest_city = city_frame.iloc[-1] if not city_frame.empty else None

    station_scope = _station_scope(stations, selected_city)

    return {
        "has_data": True,
        "cities": cities,
        "selected_city": selected_city,
        "summary_cards": [
            {
                "label": "Cities",
                "value": f"{len(cities):,}",
                "detail": "Historical bulletin files",
            },
            {
                "label": "Records",
                "value": f"{len(bulletins):,}",
                "detail": "Daily AQI observations",
            },
            {
                "label": "Average AQI",
                "value": f"{average_aqi:.0f}",
                "detail": category_for_aqi(average_aqi),
            },
            {
                "label": "Date Range",
                "value": _format_date_range(date_min, date_max),
                "detail": "Across all cities",
            },
        ],
        "selected_cards": _selected_city_cards(city_frame, latest_city),
        "trend": _trend_payload(city_frame),
        "category_items": _category_distribution(bulletins),
        "top_cities": _top_city_items(bulletins),
        "pollutant_items": _pollutant_items(bulletins),
        "station_summary": _station_summary(station_scope, selected_city),
        "station_pollutants": _station_pollutants(station_scope),
        "station_map_points": _station_map_points(stations),
    }


def _resolve_city(requested_city, cities):
    city_lookup = {city.casefold(): city for city in cities}
    if requested_city and requested_city.casefold() in city_lookup:
        return city_lookup[requested_city.casefold()]
    if "delhi" in city_lookup:
        return city_lookup["delhi"]
    return cities[0]


def _format_date(value):
    if pd.isna(value):
        return "N/A"
    return value.strftime("%d %b %Y")


def _format_date_range(start, end):
    if pd.isna(start) or pd.isna(end):
        return "N/A"
    return f"{start.strftime('%Y')} - {end.strftime('%Y')}"


def _percent(value, total):
    if not total:
        return 0
    return round((float(value) / float(total)) * 100, 1)


def _selected_city_cards(city_frame, latest_city):
    if city_frame.empty or latest_city is None:
        return []

    latest_aqi = float(latest_city["Index Value"])
    best_aqi = float(city_frame["Index Value"].min())
    worst_aqi = float(city_frame["Index Value"].max())
    dominant_pollutant = (
        city_frame["Prominent Pollutant"].replace("NAN", pd.NA).dropna().mode()
    )

    return [
        {
            "label": "Latest AQI",
            "value": f"{latest_aqi:.0f}",
            "detail": _format_date(latest_city["date"]),
            "color": color_for_aqi(latest_aqi),
        },
        {
            "label": "Best Day",
            "value": f"{best_aqi:.0f}",
            "detail": category_for_aqi(best_aqi),
            "color": color_for_aqi(best_aqi),
        },
        {
            "label": "Worst Day",
            "value": f"{worst_aqi:.0f}",
            "detail": category_for_aqi(worst_aqi),
            "color": color_for_aqi(worst_aqi),
        },
        {
            "label": "Main Pollutant",
            "value": dominant_pollutant.iloc[0] if not dominant_pollutant.empty else "N/A",
            "detail": "Most frequent driver",
            "color": "#0f766e",
        },
    ]


def _trend_payload(city_frame):
    if city_frame.empty:
        return {"values": [], "points": "", "area_points": "", "point_items": []}

    monthly = (
        city_frame.set_index("date")["Index Value"]
        .sort_index()
        .resample("MS")
        .mean()
        .dropna()
        .tail(24)
    )
    values = [round(float(value), 1) for value in monthly.tolist()]
    labels = [index.strftime("%b %Y") for index in monthly.index]
    line = _line_chart_points(values)

    return {
        "values": values,
        "labels": labels,
        "points": line["points"],
        "area_points": line["area_points"],
        "point_items": line["point_items"],
        "first_label": labels[0] if labels else "",
        "last_label": labels[-1] if labels else "",
        "min_value": f"{min(values):.0f}" if values else "N/A",
        "max_value": f"{max(values):.0f}" if values else "N/A",
        "avg_value": f"{(sum(values) / len(values)):.0f}" if values else "N/A",
    }


def _line_chart_points(values, width=720, height=260, padding=30):
    if not values:
        return {"points": "", "area_points": "", "point_items": []}

    low = min(values)
    high = max(values)
    if low == high:
        low -= 1
        high += 1

    span_x = width - (padding * 2)
    span_y = height - (padding * 2)
    bottom = height - padding
    point_items = []

    for index, value in enumerate(values):
        ratio_x = 0 if len(values) == 1 else index / (len(values) - 1)
        ratio_y = (value - low) / (high - low)
        x = padding + (ratio_x * span_x)
        y = bottom - (ratio_y * span_y)
        point_items.append(
            {
                "x": round(x, 1),
                "y": round(y, 1),
                "value": f"{value:.0f}",
                "color": color_for_aqi(value),
            }
        )

    points = " ".join(f"{point['x']},{point['y']}" for point in point_items)
    area_points = (
        f"{point_items[0]['x']},{bottom} "
        f"{points} "
        f"{point_items[-1]['x']},{bottom}"
    )

    return {"points": points, "area_points": area_points, "point_items": point_items}


def _category_distribution(frame):
    counts = frame["Air Quality"].fillna("Unknown").astype(str).str.strip().value_counts()
    total = int(counts.sum())
    items = []

    for category in CATEGORY_ORDER:
        count = int(counts.get(category, 0))
        if count:
            items.append(
                {
                    "name": category,
                    "count": f"{count:,}",
                    "pct": _percent(count, total),
                    "color": CATEGORY_COLORS[category],
                }
            )

    other_count = int(counts.drop(labels=CATEGORY_ORDER, errors="ignore").sum())
    if other_count:
        items.append(
            {
                "name": "Unknown",
                "count": f"{other_count:,}",
                "pct": _percent(other_count, total),
                "color": CATEGORY_COLORS["Unknown"],
            }
        )

    return items


def _top_city_items(frame):
    grouped = (
        frame.groupby("City", as_index=False)
        .agg(
            avg_aqi=("Index Value", "mean"),
            peak_aqi=("Index Value", "max"),
            days=("Index Value", "size"),
        )
        .query("days >= 25")
        .sort_values(["avg_aqi", "peak_aqi"], ascending=False)
        .head(10)
    )

    max_value = float(grouped["avg_aqi"].max()) if not grouped.empty else 1
    return [
        {
            "name": row["City"],
            "avg": f"{row['avg_aqi']:.0f}",
            "peak": f"{row['peak_aqi']:.0f}",
            "days": f"{int(row['days']):,}",
            "width": round((float(row["avg_aqi"]) / max_value) * 100, 1),
            "color": color_for_aqi(float(row["avg_aqi"])),
        }
        for _, row in grouped.iterrows()
    ]


def _pollutant_items(frame):
    counts = (
        frame["Prominent Pollutant"]
        .replace(["", "NAN", "NONE"], pd.NA)
        .dropna()
        .astype(str)
        .str.upper()
        .value_counts()
        .head(8)
    )
    max_count = int(counts.max()) if not counts.empty else 1

    return [
        {
            "name": name,
            "count": f"{int(count):,}",
            "width": round((int(count) / max_count) * 100, 1),
        }
        for name, count in counts.items()
    ]


def _station_scope(stations, selected_city):
    if stations.empty or "city" not in stations.columns:
        return stations

    city_scope = stations[stations["city"].str.casefold() == selected_city.casefold()]
    return city_scope if not city_scope.empty else stations


def _station_summary(station_scope, selected_city):
    if station_scope.empty:
        return {
            "scope": "No station data",
            "readings": "0",
            "avg_aqi": "N/A",
            "latest_update": "N/A",
            "weather": [],
        }

    exact_city = (
        station_scope["city"].dropna().str.casefold().eq(selected_city.casefold()).any()
    )
    scope_name = selected_city if exact_city else "All station records"

    weather = [
        {
            "label": "Temperature",
            "value": f"{station_scope['Temperature_C'].mean():.1f} C",
        },
        {"label": "Humidity", "value": f"{station_scope['Humidity_%'].mean():.1f}%"},
        {
            "label": "Wind",
            "value": f"{station_scope['Wind_Speed_kmh'].mean():.1f} km/h",
        },
    ]

    latest_update = station_scope["last_update"].max()
    latest_update_text = (
        latest_update.strftime("%d %b %Y %H:%M") if not pd.isna(latest_update) else "N/A"
    )

    avg_aqi = station_scope["AQI"].mean()
    return {
        "scope": scope_name,
        "readings": f"{len(station_scope):,}",
        "avg_aqi": f"{avg_aqi:.0f}" if not pd.isna(avg_aqi) else "N/A",
        "latest_update": latest_update_text,
        "weather": weather,
    }


def _station_pollutants(station_scope):
    if station_scope.empty:
        return []

    grouped = (
        station_scope.dropna(subset=["pollutant_avg"])
        .groupby("pollutant_id")["pollutant_avg"]
        .mean()
        .sort_values(ascending=False)
        .head(8)
    )
    max_value = float(grouped.max()) if not grouped.empty else 1

    return [
        {
            "name": str(name).upper(),
            "value": f"{float(value):.1f}",
            "width": round((float(value) / max_value) * 100, 1),
        }
        for name, value in grouped.items()
    ]


def _station_map_points(stations):
    if stations.empty:
        return []

    latest_update = stations["last_update"].max()
    snapshot = stations
    if not pd.isna(latest_update):
        snapshot = stations[stations["last_update"] == latest_update]

    grouped = (
        snapshot.dropna(subset=["latitude", "longitude", "AQI"])
        .groupby("city", as_index=False)
        .agg(
            latitude=("latitude", "mean"),
            longitude=("longitude", "mean"),
            aqi=("AQI", "mean"),
        )
        .sort_values("aqi", ascending=False)
        .head(220)
    )

    points = []
    for _, row in grouped.iterrows():
        longitude = float(row["longitude"])
        latitude = float(row["latitude"])
        aqi = float(row["aqi"])
        if not (66 <= longitude <= 100 and 5 <= latitude <= 38):
            continue

        points.append(
            {
                "city": row["city"],
                "x": round(((longitude - 66) / 34) * 100, 2),
                "y": round(((38 - latitude) / 33) * 100, 2),
                "aqi": f"{aqi:.0f}",
                "color": color_for_aqi(aqi),
                "size": round(7 + min(aqi, 300) / 45, 1),
            }
        )

    return points
