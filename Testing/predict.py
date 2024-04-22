# block warnings
import sys

sys.path.append("..")
import warnings
import pandas as pd
import os
from datetime import datetime, timedelta
import numpy as np
from joblib import load


def preprocess_data(data_path):
    """
    Preprocesses the weather data for time series forecasting.

    Parameters:
    - data_path: Path to the CSV file containing the weather data.

    Returns:
    - A DataFrame with the weather data preprocessed and ready for modeling.
    """
    weather = pd.read_csv(data_path)
    # remove "temp_10" column
    weather = weather.drop(columns=["temp_10"])
    weather.rename(
        columns={
            "TmStamp": "time",
            "temp_2": "temp",
            "speed_10": "wind_spd",
            "dir_10": "wind_dir",
        },
        inplace=True,
    )

    weather = pd.read_csv(data_path)
    weather["solar"] = weather["solar"].interpolate(method="linear")
    weather["temp"] = weather["temp"].interpolate(method="linear")
    weather["wind_spd"] = weather["wind_spd"].interpolate(method="linear")
    weather["wind_dir"] = weather["wind_dir"].interpolate(method="linear")
    weather["time"] = pd.to_datetime(weather["time"])
    weather["temp"] = round((weather["temp"] * 9 / 5 + 32), 2)
    lags = range(15, 601, 15)  # For example, every 15 minutes up to 10 hours
    for lag in lags:
        shifted_col = weather["temp"].shift(lag // 15).rename(f"temp_{lag}min_ago")
        weather = pd.concat([weather, shifted_col], axis=1)
    weather["temp_rolling_mean"] = round(weather["temp"].rolling(window=4).mean(), 2)
    weather["temp_rolling_std"] = round(weather["temp"].rolling(window=4).std(), 2)
    weather["hour_sin"] = np.sin(2 * np.pi * weather["time"].dt.hour / 24)
    weather["hour_cos"] = np.cos(2 * np.pi * weather["time"].dt.hour / 24)
    weather["month_sin"] = np.sin(2 * np.pi * weather["time"].dt.month / 12)
    weather["month_cos"] = np.cos(2 * np.pi * weather["time"].dt.month / 12)
    weather["year"] = weather["time"].dt.year
    weather["month"] = weather["time"].dt.month
    weather["day"] = weather["time"].dt.day
    weather["hour"] = weather["time"].dt.hour
    weather["target_temp"] = weather["temp"].shift(-4)
    weather = weather.dropna()
    return_val = weather.copy()
    return return_val


def predict_temp(model_name, data_path, target_variable):
    # Load and preprocess data
    data = preprocess_data(data_path)

    # Define the training features explicitly (Example based on your provided features)
    training_features = [
        "temp",
        "wind_spd",
        "wind_dir",
        "solar",
        "temp_15min_ago",
        "temp_30min_ago",
        "temp_45min_ago",
        "temp_60min_ago",
        # Continue listing all features used during training except the target variable...
        "hour_sin",
        "hour_cos",
        "month_sin",
        "month_cos",
        "year",
        "month",
        "day",
        "hour",
    ]

    # Ensure the DataFrame for prediction only contains the features used during training
    if target_variable in data.columns:
        data = data.drop(columns=[target_variable])
    last_row = data[training_features].iloc[-1:]  # Select only training features

    # Load the trained model
    model_path = os.path.join(
        "Model_Directory", f"{model_name}_{target_variable}.joblib"
    )
    model = load(model_path)

    # Predict temperature
    prediction = model.predict(last_row)

    return prediction[0]  # Return the predicted temperature


def predict_master(
    model_name, models_directory="Model_Directory", csv_path="historical_data.csv"
):
    predicted_temperature = predict_temp(model_name, csv_path, "target_temp")
    return predicted_temperature


# Example usage
warnings.filterwarnings("ignore")
historical_data = "historical_data.csv"

predicted_temp_catboost = predict_master("CatBoost")
print(f"Predicted temperature using CatBoost is: {predicted_temp_catboost} °F")

predicted_temp_xgboost = predict_master("XGBoost")
print(f"Predicted temperature using XGBoost is: {predicted_temp_xgboost} °F")

predicted_temp_randomforest = predict_master("RandomForest")
print(f"Predicted temperature using RandomForest is: {predicted_temp_randomforest} °F")
