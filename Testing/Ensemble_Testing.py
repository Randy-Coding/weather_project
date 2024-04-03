import sklearn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
)
import joblib
from joblib import dump
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import cross_val_score
import os
from sklearn.model_selection import TimeSeriesSplit


def preprocess_data(data_path):
    """
    Preprocesses the weather data for time series forecasting.

    Parameters:
    - data_path: Path to the CSV file containing the weather data.

    Returns:
    - A DataFrame with the weather data preprocessed and ready for modeling.
    """
    # Load the data
    weather = pd.read_csv(data_path)

    # Interpolate missing values linearly
    weather["solar"] = weather["solar"].interpolate(method="linear")
    weather["temp"] = weather["temp"].interpolate(method="linear")
    weather["wind_spd"] = weather["wind_spd"].interpolate(method="linear")
    weather["wind_dir"] = weather["wind_dir"].interpolate(method="linear")

    # Convert 'time' to datetime and temperatures to Fahrenheit
    weather["time"] = pd.to_datetime(weather["time"])
    weather["temp"] = round((weather["temp"] * 9 / 5 + 32), 2)

    # Create lagged features for temperature every 15 minutes up to an hour
    for i, lag in enumerate(range(15, 61, 15), start=1):
        weather[f"temp_{lag}min_ago"] = weather["temp"].shift(i)

    # Calculate rolling mean and standard deviation for temperature
    weather["temp_rolling_mean"] = round(weather["temp"].rolling(window=4).mean(), 2)
    weather["temp_rolling_std"] = round(weather["temp"].rolling(window=4).std(), 2)

    # Encode cyclical features for hour and month using sine and cosine
    weather["hour_sin"] = np.sin(2 * np.pi * weather["time"].dt.hour / 24)
    weather["hour_cos"] = np.cos(2 * np.pi * weather["time"].dt.hour / 24)
    weather["month_sin"] = np.sin(2 * np.pi * weather["time"].dt.month / 12)
    weather["month_cos"] = np.cos(2 * np.pi * weather["time"].dt.month / 12)

    # Extract additional time components
    weather["year"] = weather["time"].dt.year
    weather["month"] = weather["time"].dt.month
    weather["day"] = weather["time"].dt.day
    weather["hour"] = weather["time"].dt.hour

    # Set the target variable for temperature forecasting
    weather["target_temp"] = weather["temp"].shift(-4)
    weather["target_wind_spd"] = weather["wind_spd"].shift(-4)
    weather["target_wind_dir"] = weather["wind_dir"].shift(-4)
    weather["target_solar"] = weather["solar"].shift(-4)

    # Drop rows with NaN values resulting from shifting and rolling operations
    weather = weather.dropna()

    return weather


def test_data(model, model_name: str, data_path, cv=5):

    weather_data = preprocess_data(data_path)

    # Calculate the split index for an 80/20 split
    split_index = int(len(weather_data) * 0.8)

    # Split the data into training and testing sets based on the calculated index
    training_data = weather_data.iloc[:split_index]
    testing_data = weather_data.iloc[split_index:]

    # specify target and features for both training and testing sets
    X_train = training_data.drop(
        columns=[
            "target_temp",
            "target_wind_spd",
            "target_wind_dir",
            "target_solar",
            "time",
        ]
    )
    y_train = training_data["target_temp"]
    X_test = testing_data.drop(
        columns=[
            "target_temp",
            "target_wind_spd",
            "target_wind_dir",
            "target_solar",
            "time",
        ]
    )
    y_test = testing_data["target_temp"]

    # Define scoring metrics
    scoring = {"MAE": "neg_mean_absolute_error", "MSE": "neg_mean_squared_error"}

    # Perform cross-validation
    print("performing cross validation")
    tscv = TimeSeriesSplit(n_splits=cv)

    cv_results = cross_validate(
        model, X_train, y_train, cv=tscv, scoring=scoring, return_train_score=False
    )

    # Compute RMSE scores from MSE scores
    cv_rmse_scores = np.sqrt(-cv_results["test_MSE"])

    # Convert MAE scores to positive values
    cv_mae_scores = -cv_results["test_MAE"]

    print(f"\nCross-Validation Results for: {model_name}")
    print(f"MAE scores: {cv_mae_scores}")
    print(f"Mean MAE: {cv_mae_scores.mean()}")
    print(f"RMSE scores: {cv_rmse_scores}")
    print(f"Mean RMSE: {cv_rmse_scores.mean()}")
    print(f"Standard deviation (RMSE): {cv_rmse_scores.std()}")

    # Training the model on the full training data and evaluate on the test set
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"\n\nTest Set Results for: {model_name}")
    # Calculate MAE
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Mean Absolute Error (MAE): {mae}")
    # Calculate MSE
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error (MSE): {mse}")
    # Calculate RMSE
    rmse = np.sqrt(mse)
    print(f"Root Mean Squared Error (RMSE): {rmse}")

    full_data = pd.concat([X_train, X_test])
    full_targets = pd.concat([y_train, y_test])

    # Train the model on the full dataset
    model.fit(full_data, full_targets)

    # Save the fully trained model
    model_save_path = os.path.join("Model_Directory", f"{model_name}.joblib")
    dump(model, model_save_path)
    print(f"Model saved to {model_save_path}")
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Actual vs. Predicted Temperature")
    plt.show()


# make a gradient boosting model with some parameters
gradient_booster = GradientBoostingRegressor(n_estimators=200, random_state=42)
cat_booster = CatBoostRegressor(n_estimators=200, random_state=42, verbose=False)
random_forest = RandomForestRegressor(n_estimators=200, random_state=42)
test_data(cat_booster, "CatBoost", "Training_input.csv")
test_data(gradient_booster, "Gradient Boosting", "Training_input.csv")
test_data(random_forest, "Random Forest", "Training_input.csv")
