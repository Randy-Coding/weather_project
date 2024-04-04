import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso, Ridge, ElasticNet, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_score
import os
from joblib import dump

# import 'Training_Data.csv' and 'Testing_Data.csv'


def preprocess_data(data_path):
    """
    Preprocesses the weather data for time series forecasting.

    Parameters:
    - data_path: Path to the CSV file containing the weather data.

    Returns:
    - A DataFrame with the weather data preprocessed and ready for modeling.
    """
    weather = pd.read_csv(data_path)
    weather["solar"] = weather["solar"].interpolate(method="linear")
    weather["temp"] = weather["temp"].interpolate(method="linear")
    weather["wind_spd"] = weather["wind_spd"].interpolate(method="linear")
    weather["wind_dir"] = weather["wind_dir"].interpolate(method="linear")
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
    weather = weather.dropna()
    return_val = weather.copy()
    return return_val


def test_model(model_name: str, data_path: str, target_variable: str, cv: int = 5):
    weather_data = preprocess_data(data_path)

    # Calculate the split index for an 80/20 split
    split_index = int(len(weather_data) * 0.8)

    # Split the data into training and testing sets based on the calculated index
    training_data = weather_data.iloc[:split_index]
    testing_data = weather_data.iloc[split_index:]

    X_train = training_data.drop(
        columns=[
            "target_temp",
            "target_wind_spd",
            "target_wind_dir",
            "target_solar",
            "time",
        ]
    )
    y_train = training_data[target_variable]
    X_test = testing_data.drop(
        columns=[
            "target_temp",
            "target_wind_spd",
            "target_wind_dir",
            "target_solar",
            "time",
        ]
    )
    y_test = testing_data[target_variable]

    # Define the model
    if model_name == "LinearRegression":
        model = LinearRegression()
    elif model_name == "LassoRegression":
        model = Lasso()
    elif model_name == "RidgeRegression":
        model = Ridge()
    elif model_name == "ElasticNet":
        model = ElasticNet()

    full_data = pd.concat([X_train, X_test])
    full_targets = pd.concat([y_train, y_test])

    # Train the model on the full dataset
    model.fit(full_data, full_targets)
    # Save the trained model
    model_save_path = os.path.join(
        "Model_Directory", f"{model_name}_{target_variable}.joblib"
    )
    dump(model, model_save_path)
    print(f"Model saved to {model_save_path}")


# create a ridge regression model
test_model("LinearRegression", "Training_input.csv", "target_temp")
test_model("LinearRegression", "Training_input.csv", "target_wind_dir")
test_model("LinearRegression", "Training_input.csv", "target_wind_spd")
test_model("LinearRegression", "Training_input.csv", "target_solar")

test_model("LassoRegression", "Training_input.csv", "target_temp")
test_model("LassoRegression", "Training_input.csv", "target_wind_dir")
test_model("LassoRegression", "Training_input.csv", "target_wind_spd")
test_model("LassoRegression", "Training_input.csv", "target_solar")

test_model("RidgeRegression", "Training_input.csv", "target_temp")
test_model("RidgeRegression", "Training_input.csv", "target_wind_dir")
test_model("RidgeRegression", "Training_input.csv", "target_wind_spd")
test_model("RidgeRegression", "Training_input.csv", "target_solar")

test_model("ElasticNet", "Training_input.csv", "target_temp")
test_model("ElasticNet", "Training_input.csv", "target_wind_dir")
test_model("ElasticNet", "Training_input.csv", "target_wind_spd")
test_model("ElasticNet", "Training_input.csv", "target_solar")
