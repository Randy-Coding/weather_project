# block warnings
import warnings
import pandas as pd
import os
import joblib
from datetime import datetime, timedelta
import numpy as np


def generate_features(filepath="historical_data.csv"):
    df = pd.read_csv(filepath)

    # Assuming 'time' is a column in your CSV. Convert it to datetime if not already.
    df["time"] = pd.to_datetime(df["time"])
    df.set_index("time", inplace=True)

    variables = [
        "wind_spd",
        "wind_dir",
        "temp",
        "solar",
    ]  # Add or remove variables based on your CSV

    for variable in variables:
        for lag in [1, 2, 3, 4]:  # Assuming data is recorded every 15 minutes
            df[f"{variable}_lag_{lag*15}min"] = df[variable].shift(lag)

        df[f"{variable}_rolling_mean_60min"] = df[variable].rolling(window=4).mean()
        df[f"{variable}_rolling_std_60min"] = df[variable].rolling(window=4).std()

    if (
        "hour_sin" not in df.columns
        and "hour_cos" not in df.columns
        and "month_sin" not in df.columns
        and "month_cos" not in df.columns
    ):
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df.dropna()
    return df


def predict_temp(models_directory="Trained_Models", csv_path="historical_data.csv"):
    forecasts = []

    historical_data = pd.read_csv(csv_path)
    historical_data.rename(
        columns={
            "TmStamp": "time",  # Ensure this matches your CSV column names
            "temp_2": "temp",  # Adjust based on the column you're using for temperature
            # Include other relevant columns as needed
        },
        inplace=True,
    )
    # translate temp to farenheit
    historical_data["time"] = pd.to_datetime(historical_data["time"])
    historical_data.set_index("time", inplace=True)  # Use 'time' as the DataFrame index
    historical_data["hour"] = historical_data.index.hour
    historical_data["month"] = historical_data.index.month
    # Adjust the temperature conversion as necessary

    start_time = historical_data.index[-1] + timedelta(
        hours=1
    )  # Begin predictions from the next hour
    end_time = start_time.replace(
        hour=23, minute=0, second=0, microsecond=0
    )  # Predict until the end of the day

    current_time = start_time
    while current_time <= end_time:
        hour_month_suffix = f"{current_time.hour}-{current_time.month}"

        # Predict with ARIMA model for temperature
        temp_model_path = os.path.join(
            models_directory, f"HH-{hour_month_suffix}_temp.joblib"
        )
        if os.path.exists(temp_model_path):
            model = joblib.load(temp_model_path)
            forecast = model.get_forecast(steps=1)
            forecast_mean_temp = forecast.predicted_mean.iloc[0]
            forecasts.append(round(forecast_mean_temp, 2))

            # Update historical_data with the new temperature forecast for the next iteration
            new_row = pd.Series({"temp": forecast_mean_temp}, name=current_time)
            historical_data = historical_data._append(new_row)
        else:
            print(f"Temperature model not found for time {current_time}.")

        # Optionally predict solar radiation, without appending its forecasts to `forecasts` or `historical_data`
        solar_model_path = os.path.join(
            models_directory, f"HH-{hour_month_suffix}_solar.joblib"
        )
        if os.path.exists(solar_model_path):
            model = joblib.load(solar_model_path)
            forecast = model.get_forecast(steps=1)
            forecast_mean_solar = forecast.predicted_mean.iloc[0]
            # Here, we could do something with the solar prediction if needed but it's not appended to forecasts or historical_data

        current_time += timedelta(hours=1)

    return forecasts


# Example usage
warnings.filterwarnings("ignore")
models_directory = "Trained_Models"
csv_path = "historical_data.csv"
historical_data = "historical_data.csv"
forecast = predict_temp(models_directory, csv_path)
print(f"Forecast for the day: {forecast}")
