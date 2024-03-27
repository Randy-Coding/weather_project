import sklearn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
)
import joblib
from joblib import dump
from catboost import CatBoostRegressor
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from sklearn.model_selection import cross_val_score

# import 'Training_Data.csv' and 'Testing_Data.csv'
training_data = pd.read_csv(
    r"C:\Users\Kage\Documents\GitHub\weather_project\Testing\Training_Data.csv"
)
testing_data = pd.read_csv(
    r"C:\Users\Kage\Documents\GitHub\weather_project\Testing\Testing_Data.csv"
)


def test_data(model, model_name: str, training_data, testing_data, cv=5):
    # specify target and features for both training and testing sets
    X_train = training_data.drop(columns=["target_temp", "time"])
    y_train = training_data["target_temp"]
    X_test = testing_data.drop(columns=["target_temp", "time"])
    y_test = testing_data["target_temp"]

    # Define scoring metrics
    # scoring = {"MAE": "neg_mean_absolute_error", "MSE": "neg_mean_squared_error"}

    # Perform cross-validation
    # cv_results = cross_validate(
    #     model, X_train, y_train, cv=cv, scoring=scoring, return_train_score=False
    # )

    # Compute RMSE scores from MSE scores
    # cv_rmse_scores = np.sqrt(-cv_results["test_MSE"])

    # Convert MAE scores to positive values
    # cv_mae_scores = -cv_results["test_MAE"]

    # print(f"\nCross-Validation Results for: {model_name}")
    # print(f"MAE scores: {cv_mae_scores}")
    # print(f"Mean MAE: {cv_mae_scores.mean()}")
    # print(f"RMSE scores: {cv_rmse_scores}")
    # print(f"Mean RMSE: {cv_rmse_scores.mean()}")
    # print(f"Standard deviation (RMSE): {cv_rmse_scores.std()}")

    # Training the model on the full training data and evaluate on the test set
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    errors = y_test - y_pred

    # Round errors to nearest integer
    rounded_errors = errors.round(0).abs()

    # Find how many errors are above 20
    sum_of_errors = 0
    for error in rounded_errors:
        if error > 10:
            sum_of_errors += 1

    print(sum_of_errors)

    # print(f"\n\nTest Set Results for: {model_name}")
    # Calculate MAE
    # mae = mean_absolute_error(y_test, y_pred)
    # print(f"Mean Absolute Error (MAE): {mae}")
    # Calculate MSE
    # mse = mean_squared_error(y_test, y_pred)
    # print(f"Mean Squared Error (MSE): {mse}")
    # Calculate RMSE
    # rmse = np.sqrt(mse)
    # print(f"Root Mean Squared Error (RMSE): {rmse}")

    error_counts = rounded_errors.value_counts().sort_index()

    # Plot frequency of each error
    plt.plot(error_counts.index, error_counts.values)
    plt.title("Frequency of Rounded Errors")
    plt.xlabel("Error")
    plt.ylabel("Frequency")
    plt.show()


# create a catboost model with a depth of 10
model = CatBoostRegressor(
    iterations=200, learning_rate=0.5, depth=8, random_state=42, verbose=False
)
test_data(model, "CatBoost Regression", training_data, testing_data)

dump(model, "Catboost_Regressor.pkl")
