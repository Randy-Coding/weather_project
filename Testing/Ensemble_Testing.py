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
    scoring = {"MAE": "neg_mean_absolute_error", "MSE": "neg_mean_squared_error"}

    # Perform cross-validation
    cv_results = cross_validate(
        model, X_train, y_train, cv=cv, scoring=scoring, return_train_score=False
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


model = RandomForestRegressor(n_estimators=200, random_state=42)
test_data(model, "Random Forest Regression", training_data, testing_data)
dump(model, "Random_Forest_Model.pkl")