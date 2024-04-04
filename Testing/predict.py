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
    weather["target_temp"] = weather["temp"].shift(-4)
    weather["target_wind_spd"] = weather["wind_spd"].shift(-4)
    weather["target_wind_dir"] = weather["wind_dir"].shift(-4)
    weather["target_solar"] = weather["solar"].shift(-4)
    weather_for_printing = weather.iloc[96:]
    weather_for_printing = weather_for_printing[
        (weather["time"].dt.minute == 0) & (weather["time"].dt.second == 0)
    ]
    # make weather_for_printing equal the values after the 99th index
    #make weather equal only to values after row 40
    weather = weather[40:]


    actual_values_dict = {}

    for target in ["temp", "wind_spd", "wind_dir", "solar"]:
        # Dynamically create the list name
        # Assign the target column values to a list and store it in the dictionary
        actual_values_dict[target] = weather_for_printing[target].tolist()

    # reset index
    weather = weather.reset_index(drop=True)
    for target in ["temp", "wind_spd", "wind_dir", "solar"]:
        weather[target].iloc[104:] = np.NaN
    return weather, actual_values_dict


def update_features(features, historical_data):
    """
    Updates the feature set for the entire dataset with the latest historical values.
    
    Parameters:
    - features: DataFrame containing the feature set.
    - historical_data: DataFrame containing the historical data including latest predictions.
    """
    columns = ["temp", "wind_spd", "wind_dir", "solar"]
    lags = range(15, 601, 15)  # For example, every 15 minutes up to 10 hours

    # Update lagged features for the entire dataset
    for lag in lags:
        for col in columns:
            # Calculate the shift amount based on the time interval (assuming 15min intervals)
            shift_amount = lag // 15
            features[f"{col}_{lag}min_ago"] = features[col].shift(shift_amount)

    # Update rolling features for the entire dataset
    features["temp_rolling_mean"] = round(features["temp"].rolling(window=4).mean(), 2)
    features["temp_rolling_std"] = round(features["temp"].rolling(window=4).std(), 2)

    # Update trigonometric transformations for hour and month
    features["hour_sin"] = np.sin(2 * np.pi * historical_data["time"].dt.hour / 24)
    features["hour_cos"] = np.cos(2 * np.pi * historical_data["time"].dt.hour / 24)
    features["month_sin"] = np.sin(2 * np.pi * historical_data["time"].dt.month / 12)
    features["month_cos"] = np.cos(2 * np.pi * historical_data["time"].dt.month / 12)

    # Update wind direction transformations
    features["wind_dir_sin"] = np.sin(np.radians(features["wind_dir"]))
    features["wind_dir_cos"] = np.cos(np.radians(features["wind_dir"]))

    # Update time components
    features["year"] = historical_data["time"].dt.year
    features["month"] = historical_data["time"].dt.month
    features["day"] = historical_data["time"].dt.day
    features["hour"] = historical_data["time"].dt.hour

    

def predict_values(
    models_directory="Model_Directory",
    csv_path="historical_data.csv",
    model_type="CatBoost",
):

    result = preprocess_data(csv_path)
    historical_data = result[0]
    real_values = result[1]
    historical_data = historical_data[
        (historical_data["time"].dt.minute == 0)
        & (historical_data["time"].dt.second == 0)
    ]
    features = historical_data.drop(
        columns=[
            "time",
            "target_temp",
            "target_wind_spd",
            "target_wind_dir",
            "target_solar",
        ]
    )
    historical_data = historical_data.reset_index(drop=True)
    features = features.reset_index(drop = True)
    for i in historical_data[historical_data["temp"].isnull()].index:
        for target in ["temp", "wind_spd", "wind_dir", "solar"]:
            model_name = (f"{model_type}_target_{target}.joblib")
            model_path = os.path.join(models_directory, model_name)
            model = load(model_path)
            feature_vector = features.loc[[i - 1]]
            if pd.isnull(feature_vector["temp_15min_ago"].iloc[0]):
                update_features(features,historical_data)
            value_pred = model.predict(feature_vector)[0]
            historical_data.at[i, target] = value_pred
            features.at[i, target] = value_pred

    # put all of the temp values in a list
    pred_values = {}
    historical_data = historical_data.iloc[14:]
    for value in ["temp", "wind_spd", "wind_dir", "solar"]:
        pred_values[value] = historical_data[value].tolist()
    return historical_data, real_values, pred_values


def predict_master(
    models_directory="Model_Directory", csv_path="Predict_Here/historical_data.csv"
):

    model_name = os.path.join(f"CatBoost")
    return predict_values(models_directory, csv_path, model_name)


# Example usage
warnings.filterwarnings("ignore")
historical_data = "historical_data.csv"

result = predict_master("Model_Directory", historical_data)
real_values = result[1]
pred_values = result[2]
print(real_values["temp"])
print(pred_values["temp"])



