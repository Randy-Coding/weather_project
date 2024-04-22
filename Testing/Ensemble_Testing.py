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

    # Calculate the split index for an 80/20 split
    split_index = int(len(weather_data) * 0.8)

    # Split the data into training and testing sets based on the calculated index
    training_data = weather_data.iloc[:split_index]
    testing_data = weather_data.iloc[split_index:]

    # Specify target and features for both training and testing sets
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

    if model_name == "CatBoost":
        param_grid = {
            "n_estimators": [100, 200, 300],
            "learning_rate": [0.01, 0.05, 0.1],
            "depth": [4, 6, 8],
        }
        model = CatBoostRegressor(random_state=42, verbose=False)
    elif model_name == "XGBoost":
        # Define the parameter grid for XGBoost
        param_grid = {
            "n_estimators": [100, 500, 1000],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [3, 4, 5],
            "subsample": [0.7, 0.8, 0.9],
            "colsample_bytree": [0.7, 0.8, 0.9],
        }
        model = XGBRegressor(random_state=42, verbosity=0, early_stopping_rounds=10)

        # Split the training data for early stopping validation
        X_train_part, X_val, y_train_part, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )

        # Initialize variables to find the best model and parameters
        lowest_rmse = np.inf
        best_params = {}

        # Iterate over all combinations of parameters
        for params in ParameterGrid(param_grid):
            temp_model = model.set_params(**params)  # Set parameters
            temp_model.fit(
                X_train_part,
                y_train_part,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )

            predictions = temp_model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, predictions))

            if rmse < lowest_rmse:
                lowest_rmse = rmse
                best_model = temp_model
                best_params = params
            print(
                f"Best parameters for XGBoost: {best_params} with RMSE: {lowest_rmse}"
            )
    else:
        # Handle other models or throw an error
        raise ValueError("Unsupported model name")

    # For CatBoost and other models without early stopping, use GridSearchCV as before
    if model_name != "XGBoost":
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=TimeSeriesSplit(n_splits=5),
            scoring="neg_mean_squared_error",
            verbose=2,
        )
        grid_search.fit(X_train, y_train)
        print("Best parameters found:", grid_search.best_params_)
        best_model = grid_search.best_estimator_
        # Predict and evaluate using best_model

    model = best_model
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    print(f"\nTest Set Results for: {model_name}")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")

    if model_name == "XGBoost":
        # Reconfigure the model without early_stopping_rounds for final full training
        model.set_params(early_stopping_rounds=None)
    # Training the model on the full training data and evaluate on the test set
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"\n\nTest Set Results for: {model_name}")
    mae = mean_absolute_error(y_test, y_pred)  # Calculate MAE
    mse = mean_squared_error(y_test, y_pred)  # Calculate MSE
    rmse = np.sqrt(mse)  # Calculate RMSE
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")

    full_data = pd.concat([X_train, X_test])
    full_targets = pd.concat([y_train, y_test])

    # Train the model on the full dataset
    model.fit(full_data, full_targets)

    # Save the fully trained model
    model_save_path = os.path.join(
        "Model_Directory", f"{model_name}_{target_variable}.joblib"
    )
    dump(model, model_save_path)
    print(f"Model saved to {model_save_path}")


# make a gradient boosting model with some parameters
test_model("XGBoost", "Training_input.csv", "target_temp")
test_model("CatBoost", "Training_input.csv", "target_temp")

