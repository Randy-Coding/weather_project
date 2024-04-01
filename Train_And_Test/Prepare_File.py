import pandas as pd
import numpy as np
import os
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from math import sqrt
from joblib import dump
import warnings

# import catboost and train_Test_split
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split


warnings.filterwarnings("ignore")


def generate_features(df, variable):

    for lag in [1, 2, 3, 4]:  # Considering data is recorded every 15 minutes
        df[f"{variable}_lag_{lag*15}min"] = df[variable].shift(lag)

    # Add rolling mean and standard deviation for the past 60 minutes
    df[f"{variable}_rolling_mean_60min"] = df[variable].rolling(window=4).mean()
    df[f"{variable}_rolling_std_60min"] = df[variable].rolling(window=4).std()

    if (
        "hour_sin" not in df.columns
        and "hour_cos" not in df.columns
        and "month_sin" not in df.columns
        and "month_cos" not in df.columns
    ):
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df.dropna()
    return df


def time_series_cross_validation(df, variable, order, n_splits=3):
    """
    Performs time series cross-validation on a single variable.

    Parameters:
    - df: DataFrame containing the time series.
    - variable: The target variable for forecasting.
    - order: The (p, d, q) order of the ARIMA model.
    - n_splits: Number of splits/folds for the cross-validation.

    Returns:
    - average_rmse: The average RMSE over all folds.
    """
    tscv_rmse = []
    n_records = len(df)
    fold_size = int(n_records / (n_splits + 1))

    for i in range(n_splits):
        train_end = fold_size * (i + 1)
        test_end = train_end + fold_size

        train_data = df[variable][:train_end]
        test_data = df[variable][train_end:test_end]

        model = SARIMAX(
            train_data,
            order=order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        model_fit = model.fit(disp=False)

        predictions = model_fit.forecast(steps=len(test_data))
        rmse = sqrt(mean_squared_error(test_data, predictions))
        tscv_rmse.append(rmse)

    average_rmse = np.mean(tscv_rmse)
    return average_rmse


def generate_arima_models(input_directory, output_directory):
    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Define ranges for p, d, and q
    p_range = range(0, 3)
    d_range = range(0, 2)
    q_range = range(0, 3)

    # List all CSV files in the input directory
    csv_files = [f for f in os.listdir(input_directory) if f.endswith(".csv")]

    for file in csv_files:
        print(f"Processing {file}...")
        df = pd.read_csv(os.path.join(input_directory, file))

        # Ensure 'time' is datetime and set as index
        df["time"] = pd.to_datetime(df["time"])
        df.set_index("time", inplace=True)

        # Assuming generate_features is a function you've defined to add necessary features
        df = generate_features(df, "solar")
        df = generate_features(df, "temp")

        # Dropping columns not needed for this example, adjusting as necessary for your use case
        df.drop(columns=["wind_spd", "wind_dir"], errors="ignore", inplace=True)

        for variable in ["solar", "temp"]:
            # Create target variables by shifting
            df[f"target_{variable}"] = df[variable].shift(
                -4
            )  # Shift by 4 for 60 minutes into the future

            best_aic = np.inf
            best_order = None
            best_model = None
            best_rmse = np.inf

            for p in p_range:
                for d in d_range:
                    for q in q_range:
                        try:
                            order = (p, d, q)
                            tmp_model = SARIMAX(
                                df[variable],
                                order=order,
                                enforce_stationarity=False,
                                enforce_invertibility=False,
                            )
                            tmp_model_fit = tmp_model.fit(disp=False)

                            if tmp_model_fit.aic < best_aic:
                                best_aic = tmp_model_fit.aic
                                best_order = (p, d, q)
                                best_model = tmp_model_fit
                            rmse = time_series_cross_validation(df, variable, order)
                            if rmse < best_rmse:
                                best_rmse = rmse
                                best_order = order
                        except Exception as e:
                            print(f"Failed to fit model for order {order}: {str(e)}")
                            continue

            # Ensure no NaN values before model fitting
            df.dropna(inplace=True)

            if best_model is not None:
                # Save the best model
                model_filename = os.path.join(
                    output_directory, f"{file.replace('.csv', '')}_{variable}.joblib"
                )
                dump(best_model, model_filename)
                print(f"Best Model saved for {file} {variable} with order {best_order}")
            else:
                print(f"No suitable model found for {file} {variable}.")


def generate_catboost_models(input_directory, output_directory):

    os.makedirs(output_directory, exist_ok=True)
    csv_files = [f for f in os.listdir(input_directory) if f.endswith(".csv")]
    for file in csv_files:
        print(f"Processing {file}...")
        df = pd.read_csv(os.path.join(input_directory, file))
        df["time"] = pd.to_datetime(df["time"])
        df.set_index("time", inplace=True)
        df = generate_features(df, "wind_spd")
        df = generate_features(df, "wind_dir")
        df.drop(columns=["temp", "solar"], errors="ignore", inplace=True)
        for variable in ["wind_spd", "wind_dir"]:
            df[f"target_{variable}"] = df[variable].shift(-4)
            df.dropna(inplace=True)
            X = df.drop(columns=[f"target_{variable}", variable])
            y = df[f"target_{variable}"]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, shuffle=False
            )
            model = CatBoostRegressor(
                iterations=200, learning_rate=0.1, depth=6, verbose=False
            )
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            rmse = sqrt(mean_squared_error(y_test, predictions))
            model_filename = os.path.join(
                output_directory, f"{file.replace('.csv', '')}_{variable}.joblib"
            )
            model.save_model(model_filename)
            print(f"Best Model saved for {file} {variable} with RMSE {rmse}")


def generate_models(data_directory, output_directory):
    generate_catboost_models(data_directory, output_directory)


# Directory paths
data_directory = "CSV_Collection"
output_directory = "Trained_Models"

# Ensure the output directory exists
os.makedirs(output_directory, exist_ok=True)

# Call the function
generate_models(data_directory, output_directory)
