import pandas as pd
import os
import joblib
from datetime import datetime, timedelta


def update_forecast(models_directory="Trained_Models", csv_path="weather_data.csv"):
    """
    Reads the ever-updating CSV, uses the data up to the current available hour for predictions,
    and dynamically updates future forecasts based on new and predicted data for a single day.

    Parameters:
    - models_directory: The directory where the models are saved.
    - csv_path: Path to the CSV file with weather measurements.
    """
    forecasts = []  # Initial forecast list

    data = pd.read_csv(csv_path)
    data["time"] = pd.to_datetime(data["time"])  # Convert time column to datetime

    start_time = data["time"].min()  # Get the start time for the day's predictions
    end_time = start_time.replace(
        hour=23, minute=0, second=0, microsecond=0
    )  # End time is always 23:00 of the same day

    current_time = start_time

    while current_time <= end_time:
        # Check if data for the current time exists or if we need to use forecasted data
        if current_time in data["time"].values:
            closest_time = data[data["time"] == current_time].iloc[-1]
        else:
            closest_time = forecasts[
                -1
            ]  # Use the last forecasted data if no actual data is available

        month = current_time.month
        hour = current_time.hour

        model_filename = f"model_{hour}_{month}.pkl"
        model_path = os.path.join(models_directory, model_filename)

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"No model found for hour {hour} and month {month}."
            )

        model = joblib.load(model_path)

        # Prepare the input for prediction
        feature_names = [
            "temp",
            "wind_spd",
            "wind_dir",
            "solar",
            "hour",
            "month",
        ]  # Assuming these are the features your model was trained on
        # Assuming closest_time is a Series with the latest available data
        input_features = {feature: [closest_time[feature]] for feature in feature_names}
        input_df = pd.DataFrame.from_dict(input_features)

        forecast = model.predict(input_df)[0]

        # Append the forecast to the data DataFrame for use in future iterations
        forecast_data = {
            "time": current_time,
            "temp": forecast[0],
            "wind_spd": forecast[1],
            "wind_dir": forecast[2],
            "solar": forecast[3],
            "hour": hour,
            "month": month,
        }
        data = data._append(forecast_data, ignore_index=True)
        forecasts.append(forecast_data)  # Store the forecast for output

        # Increment current_time by one hour for the next iteration
        current_time += timedelta(hours=1)

    # Extract only the predicted temperatures for the return value, if needed
    predicted_temps = [round(forecast["temp"], 2) for forecast in forecasts]
    return predicted_temps


# Example usage
models_directory = "Trained_Models"
csv_path = "processed.csv"
forecast = update_forecast(models_directory, csv_path)
print(f"Forecast for the day: {forecast}")
