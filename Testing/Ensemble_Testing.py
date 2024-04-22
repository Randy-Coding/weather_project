import sklearn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, train_test_split
from xgboost import XGBRegressor
import joblib
from joblib import dump
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import cross_val_score
import os
from sklearn.model_selection import ParameterGrid
import psutil
import time


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


def test_model(model_name: str, data_path, target_variable: str, cv=5):
    weather_data = preprocess_data(data_path)
    weather_data.sort_values("time", inplace=True)  # Ensure data is sorted by time

    # Calculate the split index for an 80/20 split
    split_index = int(len(weather_data) * 0.8)
    training_data = weather_data.iloc[:split_index]
    testing_data = weather_data.iloc[split_index:]

    # Define features and target for both training and testing sets
    X_train = training_data.drop(columns=["target_temp", "time"])
    y_train = training_data[target_variable]
    X_test = testing_data.drop(columns=["target_temp", "time"])
    y_test = testing_data[target_variable]

    if model_name == "CatBoost":
        # Use pre-determined optimal parameters
        best_params = {"depth": 8, "learning_rate": 0.1, "n_estimators": 300}
        best_model = CatBoostRegressor(**best_params, random_state=42, verbose=False)
    elif model_name == "XGBoost":
        best_params = {
            "colsample_bytree": 0.8,
            "learning_rate": 0.1,
            "max_depth": 5,
            "n_estimators": 1000,
            "subsample": 0.9,
            "verbosity": 0,
            "random_state": 42,
            "early_stopping_rounds": 10,  # Set in the constructor
        }
        best_model = XGBRegressor(**best_params)
    elif model_name == "RandomForest":
        # Default parameters for RandomForestRegressor, adjust as necessary
        best_params = {
            "n_estimators": 100,
            "max_depth": None,  # None means the nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples
            "random_state": 42,
        }
        best_model = RandomForestRegressor(**best_params)
    else:
        raise ValueError("Unsupported model name")

    model = best_model
    print("Training now")
    start_time = time.time()
    process = psutil.Process(os.getpid())
    initial_memory_use = process.memory_info().rss

    # Fit the model and include an evaluation set if it's XGBoost
    if model_name == "XGBoost":
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    else:
        model.fit(X_train, y_train)

    end_time = time.time()
    final_memory_use = process.memory_info().rss
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(
        f"Memory used for training: {(final_memory_use - initial_memory_use) / (1024 ** 2):.2f} MB"
    )
    print(f"Training and prediction time: {end_time - start_time:.2f} seconds")

    # Save the fully trained model
    model_save_path = os.path.join(
        "Model_Directory", f"{model_name}_{target_variable}.joblib"
    )
    dump(model, model_save_path)
    print(f"Model saved to {model_save_path}")

    # Plot predictions against actual values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title(f"Prediction vs Actual Value Scatter Plot for {model_name}")
    plt.grid(True)
    plt.show()


# make a gradient boosting model with some parameters
test_model("RandomForest", "Training_input.csv", "target_temp")
test_model("XGBoost", "Training_input.csv", "target_temp")
test_model("CatBoost", "Training_input.csv", "target_temp")
