from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.multioutput import MultiOutputRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import TimeSeriesSplit

# import linear regression
from sklearn.linear_model import LinearRegression

# import random forest
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
import os
import joblib


def train_models(data_directory, output_directory):
    # Define the number of splits for cross-validation
    n_splits = 5
    tscv = TimeSeriesSplit(n_splits=n_splits)

    for hour in range(24):
        for month in range(1, 13):
            file_path = os.path.join(data_directory, f"HH-{hour}-{month}.csv")
            if os.path.exists(file_path):
                data = pd.read_csv(file_path)
                data.sort_values("time", inplace=True)

                # Create rolling window features
                window_sizes = [
                    2,
                    3,
                    4,
                ]  # Corresponding to 30, 45, and 60 minutes if data is recorded every 15 minutes
                for window in window_sizes:
                    data[f"temp_roll_mean_{window}"] = (
                        data["temp"].rolling(window=window).mean()
                    )
                    data[f"wind_spd_roll_mean_{window}"] = (
                        data["wind_spd"].rolling(window=window).mean()
                    )
                    data[f"wind_dir_roll_mean_{window}"] = (
                        data["wind_dir"].rolling(window=window).mean()
                    )
                    data[f"solar_roll_mean_{window}"] = (
                        data["solar"].rolling(window=window).mean()
                    )

                    data[f"temp_roll_std_{window}"] = (
                        data["temp"].rolling(window=window).std()
                    )
                    data[f"wind_spd_roll_std_{window}"] = (
                        data["wind_spd"].rolling(window=window).std()
                    )
                    data[f"wind_dir_roll_std_{window}"] = (
                        data["wind_dir"].rolling(window=window).std()
                    )
                    data[f"solar_roll_std_{window}"] = (
                        data["solar"].rolling(window=window).std()
                    )

                # Create lagged features
                for lag in range(1, 5):  # 1 to 4 steps back, covering 15 to 60 minutes
                    data[f"temp_lag{lag}"] = data["temp"].shift(lag)
                    data[f"wind_spd_lag{lag}"] = data["wind_spd"].shift(lag)
                    data[f"wind_dir_lag{lag}"] = data["wind_dir"].shift(lag)
                    data[f"solar_lag{lag}"] = data["solar"].shift(lag)
                # Drop rows with NaN values created by shifting and rolling

                data["hour_sin"] = np.sin(2 * np.pi * data["hour"] / 24)
                data["hour_cos"] = np.cos(2 * np.pi * data["hour"] / 24)
                data["month_sin"] = np.sin(2 * np.pi * data["month"] / 12)
                data["month_cos"] = np.cos(2 * np.pi * data["month"] / 12)
                data.dropna(inplace=True)

                X = data.drop(
                    columns=[
                        "time",
                        "target_temp",
                        "target_wind_spd",
                        "target_wind_dir",
                        "target_solar",
                    ]
                )
                y = data[
                    [
                        "target_temp",
                        "target_wind_spd",
                        "target_wind_dir",
                        "target_solar",
                    ]
                ]

                # Initialize accumulators for MSE and MAE across folds
                fold_mse = []
                fold_mae = []

                for train_index, test_index in tscv.split(X):
                    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                    model = RandomForestRegressor(n_estimators=100)
                    model.fit(X_train, y_train)

                    y_pred = model.predict(X_test)

                    mse = mean_squared_error(
                        y_test, y_pred, multioutput="raw_values"
                    ).mean()
                    mae = mean_absolute_error(
                        y_test, y_pred, multioutput="raw_values"
                    ).mean()

                    fold_mse.append(mse)
                    fold_mae.append(mae)

                # Calculate the average MSE and MAE across all folds
                avg_mse = np.mean(fold_mse)
                avg_mae = np.mean(fold_mae)

                print(
                    f"Model for hour {hour}, month {month} - Average MSE: {avg_mse}, Average MAE: {avg_mae}"
                )

                # Optionally, you can re-train the model on the entire dataset
                model.fit(X, y)

                # Save the re-trained model
                model_filename = f"model_{hour}_{month}.pkl"
                joblib.dump(model, os.path.join(output_directory, model_filename))


# Directory paths
data_directory = "CSV_Collection"
output_directory = "Trained_Models"

# Ensure the output directory exists
os.makedirs(output_directory, exist_ok=True)

# Call the function
train_models(data_directory, output_directory)
