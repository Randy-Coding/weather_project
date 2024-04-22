import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso, Ridge, ElasticNet, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_score
import os
from joblib import dump
import time
import psutil

# import time series
from sklearn.model_selection import TimeSeriesSplit


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


def test_model(model_name: str, data_path: str, target_variable: str, cv: int = 5):
    weather_data = preprocess_data(data_path)
    weather_data.sort_values("time", inplace=True)  # Ensure data is sorted by time

    # Calculate the split index for an 80/20 split
    split_index = int(len(weather_data) * 0.8)
    training_data = weather_data.iloc[:split_index]
    testing_data = weather_data.iloc[split_index:]

    X_train = training_data.drop(columns=["target_temp", "time"])
    y_train = training_data[target_variable]
    X_test = testing_data.drop(columns=["target_temp", "time"])
    y_test = testing_data[target_variable]

    # Define the model
    model = {
        "LinearRegression": LinearRegression(),
        "LassoRegression": Lasso(),
        "RidgeRegression": Ridge(),
        "ElasticNet": ElasticNet(),
    }.get(model_name)

    # Initialize time series cross-validation
    tscv = TimeSeriesSplit(n_splits=cv)
    maes = []  # List to store mean absolute errors for each fold

    # Monitor initial system resources
    process = psutil.Process(os.getpid())
    initial_memory_use = process.memory_info().rss

    # Measure the training time
    start_time = time.time()

    for train_index, test_index in tscv.split(X_train):
        X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
        y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]
        model.fit(X_train_fold, y_train_fold)
        predictions_fold = model.predict(X_test_fold)
        mae_fold = mean_absolute_error(y_test_fold, predictions_fold)
        maes.append(mae_fold)

    # Train the final model on the entire training set
    model.fit(X_train, y_train)

    # Record end time and system resources after training
    end_time = time.time()
    final_memory_use = process.memory_info().rss

    # Save the trained model
    model_save_path = os.path.join(
        "Model_Directory", f"{model_name}_{target_variable}.joblib"
    )
    dump(model, model_save_path)

    # Predictions on test data and calculate MAE
    predictions = model.predict(X_test)
    final_mae = mean_absolute_error(y_test, predictions)

    # Plot the predictions against the actual values

    print(f"Training and prediction time: {end_time - start_time:.2f} seconds")
    print(
        f"Average Mean Absolute Error (MAE) across folds: {sum(maes) / len(maes):.2f}"
    )
    print(f"Final Mean Absolute Error (MAE) on test data: {final_mae:.2f}")
    print(
        f"Memory used for training: {(final_memory_use - initial_memory_use) / (1024 ** 2):.2f} MB"
    )
    print(f"Model saved to {model_save_path}\n\n\n\n")


# create a ridge regression model
test_model("ElasticNet", "Training_input.csv", "target_temp")
test_model("LinearRegression", "Training_input.csv", "target_temp")
test_model("LassoRegression", "Training_input.csv", "target_temp")
test_model("RidgeRegression", "Training_input.csv", "target_temp")
