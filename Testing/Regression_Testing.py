import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso, Ridge, ElasticNet, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


# import 'Training_Data.csv' and 'Testing_Data.csv'
training = pd.read_csv(r"Training_Data.csv")
testing = pd.read_csv(r"Testing_Data.csv")


def test_data(model, print_name: str):
    # specify target and features
    X_train = training.drop(columns=["target_temp", "time"])
    y_train = training["target_temp"]
    X_test = testing.drop(columns=["target_temp", "time"])
    y_test = testing["target_temp"]

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"\n\n\n\nResults of: {print_name}")
    # Calculate MAE
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Mean Absolute Error (MAE): {mae}")
    # Calculate MSE
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error (MSE): {mse}")
    # Calculate RMSE
    rmse = np.sqrt(mse)
    print(f"Root Mean Squared Error (RMSE): {rmse}")

    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Actual vs. Predicted Temperature")
    plt.show()


# create a ridge regression model
model = ElasticNet(alpha=0.1)
test_data(model, "ElasticNet Regression")

model = Lasso(alpha=0.1)
test_data(model, "Lasso Regression")

model = Ridge(alpha=0.1)
test_data(model, "Ridge Regression")

model = LinearRegression()
test_data(model, "Linear Regression")
