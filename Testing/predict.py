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


def predict_temp(model_name, temp_now, wind_spd_now, wind_dir_now, solar_now, temp_1h_ago):
    # Load the model
    model_path = os.path.join("Model_Directory", f"{model_name}_target_temp.joblib")
    model = load(model_path)
    
    # Create a DataFrame with the input data
    # Assuming the same transformations (e.g., temperature conversion) were applied as in training
    input_data = {
        "temp": [round((temp_now * 9 / 5 + 32), 2)],  # Convert to Fahrenheit if training was in Fahrenheit
        "wind_spd": [wind_spd_now],
        "wind_dir": [wind_dir_now],
        "solar": [solar_now],
        "temp_1h_ago": [round((temp_1h_ago * 9 / 5 + 32), 2)]  # Convert to Fahrenheit if needed
    }
    input_df = pd.DataFrame(input_data)
    
    # Predict temperature
    prediction = model.predict(input_df)
    return prediction[0]  # Return the predicted temperature
    


def predict_master(
    input_name
):
    predicted_temp = predict_temp(
    model_name=input_name,
    temp_now=7.1,  # Current temperature
    wind_spd_now=3.6,  # Current wind speed
    wind_dir_now=288.7,  # Current wind direction
    solar_now=451.2,  # Current solar radiation
    temp_1h_ago=7.1  # Temperature 1 hour ago
)
    return predicted_temp
    


# Example usage


predicted_temp_catboost = predict_master("CatBoost")
print(f"Predicted temperature using CatBoost is: {predicted_temp_catboost} °F")

predicted_temp_xgboost = predict_master("XGBoost")
print(f"Predicted temperature using XGBoost is: {predicted_temp_xgboost} °F")

predicted_temp_LassoRegression = predict_master("LassoRegression")
print(f"Predicted temperature using LassoRegression is: {predicted_temp_LassoRegression} °F")

predicted_temp_LinearRegression = predict_master("LinearRegression")
print(f"Predicted temperature using LinearRegression is: {predicted_temp_LinearRegression} °F")

predicted_temp_RidgeRegression = predict_master("RidgeRegression")
print(f"Predicted temperature using RidgeRegression is: {predicted_temp_RidgeRegression} °F")

predicted_temp_ElasticNet = predict_master("ElasticNet")
print(f"Predicted temperature using ElasticNet is: {predicted_temp_ElasticNet} °F")

