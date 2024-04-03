# block warnings
import warnings
import pandas as pd
import os
from datetime import datetime, timedelta
import numpy as np
from joblib import load


def preprocess_data(data_path, target):
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
    weather["time"] = pd.to_datetime(weather["time"])
    weather["temp"] = round((weather["temp"] * 9 / 5 + 32), 2)
    lags = range(15, 601, 15)  # For example, every 15 minutes up to 10 hours
    lagged_cols = []
    for lag in lags:
        for col in ["wind_dir", "temp", "wind_spd", "solar"]:
            # Shift the column and rename appropriately
            shifted_col = weather[col].shift(lag // 15).rename(f"{col}_{lag}min_ago")
            lagged_cols.append(shifted_col)
    # Concatenate all lagged columns alongside the original DataFrame
    weather = pd.concat([weather] + lagged_cols, axis=1)
    weather["temp_rolling_mean"] = round(weather["temp"].rolling(window=4).mean(), 2)
    weather["temp_rolling_std"] = round(weather["temp"].rolling(window=4).std(), 2)
    weather["hour_sin"] = np.sin(2 * np.pi * weather["time"].dt.hour / 24)
    weather["hour_cos"] = np.cos(2 * np.pi * weather["time"].dt.hour / 24)
    weather["month_sin"] = np.sin(2 * np.pi * weather["time"].dt.month / 12)
    weather["month_cos"] = np.cos(2 * np.pi * weather["time"].dt.month / 12)
    weather["wind_dir_sin"] = np.sin(np.radians(weather["wind_dir"]))
    weather["wind_dir_cos"] = np.cos(np.radians(weather["wind_dir"]))
    weather["year"] = weather["time"].dt.year
    weather["month"] = weather["time"].dt.month
    weather["day"] = weather["time"].dt.day
    weather["hour"] = weather["time"].dt.hour
    weather["target_wind_spd"] = weather["wind_spd"].shift(-4)
    weather["target_wind_dir"] = weather["wind_dir"].shift(-4)
    weather["target_solar"] = weather["solar"].shift(-4)
    weather["target_temp"] = weather["temp"].shift(-4)
    weather = weather.dropna()
    weather_for_printing = weather[
        (weather["time"].dt.minute == 0) & (weather["time"].dt.second == 0)
    ]
    # put all of the temp values in a list
    values = weather_for_printing[target].tolist()
    weather["temp","wind_dir","wind_spd","solar"].iloc[8:] = np.NaN
    return weather, values


def predict_values(
    models_directory="Model_Directory",
    csv_path="historical_data.csv",
    model_name="CatBoost_target_temp.joblib",
    target="solar",
):

    historical_data = preprocess_data(csv_path, target)[0]
    real_values = preprocess_data(csv_path, target)[1]
    model_path = os.path.join(models_directory, model_name)
    print(model_path)
    model = load(model_path)
    historical_data = historical_data[
        (historical_data["time"].dt.minute == 0)
        & (historical_data["time"].dt.second == 0)
    ]
    historical_data = historical_data.reset_index(drop=True)
    features = historical_data.drop(
        columns=[
            "time",
            "target_temp",
            "target_wind_spd",
            "target_wind_dir",
            "target_solar",
        ]
    )
    for i in historical_data[historical_data[target].isnull()].index:
        feature_vector = features.loc[[i - 1]]
        temp_pred = model.predict(feature_vector)[0]
        historical_data.at[i, target] = temp_pred
        features.at[i, target] = temp_pred
    # put all of the temp values in a list
    pred_temps = historical_data[target].tolist()
    return historical_data, real_values, pred_temps


def predict_master(models_directory="Model_Directory", csv_path="historical_data.csv"):

    model_name = os.path.join("XGBoost_target_solar.joblib")
    return predict_values(models_directory, csv_path, model_name, "solar")


# Example usage
warnings.filterwarnings("ignore")
historical_data = "historical_data.csv"
print(predict_master(models_directory="Model_Directory", csv_path=historical_data)[1])
print(predict_master(models_directory="Model_Directory", csv_path=historical_data)[2])
