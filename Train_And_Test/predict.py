import pandas as pd
import os
import joblib
from datetime import datetime, timedelta
import numpy as np


def generate_features_for_prediction(historical_data):
    """
    Generates necessary features for a single prediction, given
    the current weather conditions and a DataFrame of historical data
    sorted in ascending order (oldest first).

    :param historical_data: A DataFrame containing historical data to compute lagged and rolling features.
    :return: A DataFrame row (as a single-row DataFrame) with all features required for prediction.
    """
    # Ensure the historical data is a DataFrame

    window_sizes = [
        2,
        3,
        4,
    ]  # Corresponding to 30, 45, and 60 minutes if data is recorded every 15 minutes
    for window in window_sizes:
        # Create rolling features
        for window in window_sizes:
            historical_data[f"temp_roll_mean_{window}"] = (
                historical_data["temp"].rolling(window=window).mean()
            )
            historical_data[f"wind_spd_roll_mean_{window}"] = (
                historical_data["wind_spd"].rolling(window=window).mean()
            )
            historical_data[f"wind_dir_roll_mean_{window}"] = (
                historical_data["wind_dir"].rolling(window=window).mean()
            )
            historical_data[f"solar_roll_mean_{window}"] = (
                historical_data["solar"].rolling(window=window).mean()
            )

            historical_data[f"temp_roll_std_{window}"] = (
                historical_data["temp"].rolling(window=window).std()
            )
            historical_data[f"wind_spd_roll_std_{window}"] = (
                historical_data["wind_spd"].rolling(window=window).std()
            )
            historical_data[f"wind_dir_roll_std_{window}"] = (
                historical_data["wind_dir"].rolling(window=window).std()
            )
            historical_data[f"solar_roll_std_{window}"] = (
                historical_data["solar"].rolling(window=window).std()
            )

        # Create lagged features
        for lag in range(1, 5):  # 1 to 4 steps back, covering 15 to 60 minutes
            historical_data[f"temp_lag{lag}"] = historical_data["temp"].shift(lag)
            historical_data[f"wind_spd_lag{lag}"] = historical_data["wind_spd"].shift(
                lag
            )
            historical_data[f"wind_dir_lag{lag}"] = historical_data["wind_dir"].shift(
                lag
            )
            historical_data[f"solar_lag{lag}"] = historical_data["solar"].shift(lag)
        # Drop rows with NaN values created by shifting and rolling

        historical_data["hour_sin"] = np.sin(2 * np.pi * historical_data["hour"] / 24)
        historical_data["hour_cos"] = np.cos(2 * np.pi * historical_data["hour"] / 24)
        historical_data["month_sin"] = np.sin(2 * np.pi * historical_data["month"] / 12)
        historical_data["month_cos"] = np.cos(2 * np.pi * historical_data["month"] / 12)
        historical_data.dropna(inplace=True)
    # Make features_df the last row of historical_data
    features_df = historical_data.iloc[-1:]
    features_df = pd.DataFrame(features_df)
    return features_df


def predict_temp(models_directory="Trained_Models", csv_path="historical_data.csv"):
    forecasts = []

    historical_data = pd.read_csv(csv_path)
    historical_data = pd.read_csv(csv_path)
    historical_data.rename(
        columns={
            "TmStamp": "time",
            "temp_2": "temp",
            "speed_10": "wind_spd",
            "dir_10": "wind_dir",
        },
        inplace=True,
    )
    historical_data.drop(["temp_10"], axis=1, inplace=True)
    historical_data["time"] = pd.to_datetime(historical_data["time"])
    historical_data["hour"] = historical_data["time"].dt.hour
    historical_data["month"] = historical_data["time"].dt.month
    historical_data["temp"] = historical_data["temp"] * 9 / 5 + 32

    start_time = historical_data["time"].iloc[-1] + timedelta(
        hours=1
    )  # Start from the next hour
    end_time = start_time.replace(hour=23, minute=0, second=0, microsecond=0)

    current_time = start_time

    while current_time <= end_time:
        # Generate features for current prediction
        features_df = generate_features_for_prediction(historical_data.copy())

        # Check if features_df is not empty and contains expected features
        if not features_df.empty and "temp_roll_mean_2" in features_df.columns:
            model_filename = f"model_{current_time.hour}_{current_time.month}.pkl"
            model_path = os.path.join(models_directory, model_filename)

            if os.path.exists(model_path):
                model = joblib.load(model_path)
                # Predict
                forecast = model.predict(
                    features_df.drop(["time"], axis=1, errors="ignore")
                )
                forecast_data = {
                    "time": current_time,
                    "temp": forecast[0][0],
                    "wind_spd": forecast[0][1],
                    "wind_dir": forecast[0][2],
                    "solar": forecast[0][3],
                    "hour": current_time.hour,
                    "month": current_time.month,
                }

                forecasts.append(forecast_data)
                # Append forecast to historical_data for next iteration
                new_row = pd.DataFrame([forecast_data])
                historical_data = pd.concat(
                    [historical_data, new_row], ignore_index=True
                )
                historical_data["time"] = pd.to_datetime(historical_data["time"])
            else:
                print(f"Model not found for {current_time}. Skipping.")
        else:
            print(f"No data available for prediction at {current_time}. Skipping.")
            break  # Consider breaking if no data is available to avoid empty loops

        current_time += timedelta(hours=1)

    current_time += timedelta(hours=1)

    predicted_temps = [round(f["temp"], 2) for f in forecasts]
    return predicted_temps


# Example usage
models_directory = "Trained_Models"
csv_path = "historical_data.csv"
historical_data = "historical_data.csv"
forecast = predict_temp(models_directory, csv_path)
print(f"Forecast for the day: {forecast}")
