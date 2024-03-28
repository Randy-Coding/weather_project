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
