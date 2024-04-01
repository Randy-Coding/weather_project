import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from math import sqrt
import itertools
import os


def preprocess_data(data_directory):
    df = pd.read_csv(data_directory)
    df["time"] = pd.to_datetime(df["time"])
    df.set_index("time", inplace=True)
    df.index = pd.DatetimeIndex(df.index).to_period("15min")

    # Add features
    df["hour"] = df.index.hour
    df["month"] = df.index.month
    df["target_temp"] = df["temp"].shift(-4)
    df["target_wind_spd"] = df["wind_spd"].shift(-4)
    df["target_wind_dir"] = df["wind_dir"].shift(-4)
    df["target_solar"] = df["solar"].shift(-4)

    # Rolling means
    df["rolling_mean_temp"] = df["temp"].rolling(window=4).mean()
    df["rolling_mean_wind_spd"] = df["wind_spd"].rolling(window=4).mean()
    df["rolling_mean_wind_dir"] = df["wind_dir"].rolling(window=4).mean()
    df["rolling_mean_solar"] = df["solar"].rolling(window=4).mean()

    # Trigonometric features for cyclical nature of time
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # Lagging features
    for variable in ["temp", "wind_spd", "wind_dir", "solar"]:
        for lag in [1, 2, 3, 4]:
            df[f"{variable}_lag_{lag*15}min"] = df[variable].shift(lag)

    # Drop rows with NaN values
    df.dropna(inplace=True)

    # Downcast numeric columns to reduce memory usage
    for col in df.select_dtypes(include=["float64", "int64"]).columns:
        df[col] = pd.to_numeric(
            df[col], downcast="float" if df[col].dtype == "float64" else "integer"
        )

    return df


def train_model(
    data_directory,
    target_columns=[
        "target_temp",
        "target_wind_spd",
        "target_wind_dir",
        "target_solar",
    ],
    exog_columns=None,
):
    """
    Trains SARIMAX models with hyperparameter tuning for multiple target variables, using time series splitting.

    Parameters:
    - data_directory: The directory containing the dataset file.
    - target_columns: List of the names of the target variable columns.
    - exog_columns: List of names of exogenous variable columns.

    Returns:
    - Dictionary with the best model, its parameters, and test RMSE for each target.
    """
    df = preprocess_data(data_directory)
    split_ratio = 0.8
    split_index = int(len(df) * split_ratio)
    df_train = df.iloc[:split_index]
    df_test = df.iloc[split_index:]

    models_results = {}
    for col in df.columns:
        if df[col].dtype == np.float64:
            df[col] = pd.to_numeric(df[col], downcast="float")
        if df[col].dtype == np.int64:
            df[col] = pd.to_numeric(df[col], downcast="integer")
    p = d = q = range(0, 2)  # Define parameter space for SARIMAX orders
    pdq_combinations = list(itertools.product(p, d, q))


    for target in target_columns:
        best_rmse = float("inf")
        best_model = None
        best_params = None

        y_train = df_train[target]
        y_test = df_test[target]
        exog_train = df_train[exog_columns] if exog_columns else None
        exog_test = df_test[exog_columns] if exog_columns else None  # Corrected line

        for param in pdq_combinations:
                try:
                    model = SARIMAX(
                        y_train,
                        exog=exog_train,
                        order=param,
                        seasonal_order=(1, 1, 1, 96),
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    )
                    results = model.fit(disp=0)  # Set disp=0 to reduce output verbosity
                    predictions = results.get_forecast(
                        steps=len(df_test), exog=exog_test
                    ).predicted_mean
                    rmse = sqrt(mean_squared_error(y_test, predictions))

                    if rmse < best_rmse:
                        best_rmse = rmse
                        best_model = results
                        best_params = {"order": param}

                except Exception as e:
                    print(f"An error occurred: {e}")
                    continue

        print(f"Best RMSE for {target}: {best_rmse}")
        print(f"Best Parameters for {target}: {best_params}")

        models_results[target] = {
            "model": best_model,
            "rmse": best_rmse,
            "params": best_params,
        }

    return models_results


# Directory paths
data_directory = r"C:\Users\me\Documents\GitHub\weatherProject\Train_And_Test\Training_Input.csv"
output_directory = r"C:\Users\me\Documents\GitHub\weatherProject\Train_And_Test\Trained_Models"

# Ensure the output directory exists
os.makedirs(output_directory, exist_ok=True)

# Call the function
train_model(data_directory)
