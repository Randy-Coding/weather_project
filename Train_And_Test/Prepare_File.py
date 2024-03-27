from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
import numpy as np
import pandas as pd
import os
import joblib


def train_models_with_cv(data_directory, output_directory):
    n_splits = 5  # Number of splits for cross-validation
    tscv = TimeSeriesSplit(n_splits=n_splits)
    total_average_mae = []
    total_average_mse = []

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

                # Wrap the GradientBoostingRegressor with MultiOutputRegressor
                model = MultiOutputRegressor(
                    GradientBoostingRegressor(
                        n_estimators=100, max_depth=5, learning_rate=0.1
                    )
                )

                mse_scores = []
                mae_scores = []

                # Manually perform cross-validation
                for train_index, test_index in tscv.split(X):
                    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                    # Train the model on this fold
                    model.fit(X_train, y_train)

                    # Make predictions on the test set
                    y_pred = model.predict(X_test)

                    # Calculate MSE and MAE for this fold
                    mse = mean_squared_error(
                        y_test, y_pred, multioutput="raw_values"
                    ).mean()
                    mae = mean_absolute_error(
                        y_test, y_pred, multioutput="raw_values"
                    ).mean()

                    mse_scores.append(mse)
                    mae_scores.append(mae)

                average_mse = np.mean(mse_scores)
                average_mae = np.mean(mae_scores)

                print(
                    f"Model for hour {hour}, month {month} - Average MSE: {average_mse}, Average MAE: {average_mae}"
                )

                total_average_mse.append(average_mse)
                total_average_mae.append(average_mae)

                # Train the model on the entire dataset after evaluation
                model.fit(X, y)

                # Save the model
                model_filename = f"model_{hour}_{month}.pkl"
                joblib.dump(model, os.path.join(output_directory, model_filename))

    # Print overall averages
    print(f"Overall Average MAE: {np.mean(total_average_mae)}")
    print(f"Overall Average MSE: {np.mean(total_average_mse)}")


# Directory paths
data_directory = "CSV_Collection"
output_directory = "Trained_Models"

# Ensure the output directory exists
os.makedirs(output_directory, exist_ok=True)

# Call the function
train_models_with_cv(data_directory, output_directory)
