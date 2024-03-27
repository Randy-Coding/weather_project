# import the necessary libraries
import pandas as pd
import os
import joblib

# import gradient boosting regressor
from sklearn.ensemble import GradientBoostingRegressor


def train_models(model, data_directory, output_directory):
    count = 0
    for hour in range(24):
        for month in range(1, 13):
            file_path = os.path.join(data_directory, f"HH-{hour}-{month}.csv")
            if os.path.exists(file_path):
                # Read the specific hour-month data file
                data = pd.read_csv(file_path)
                # Split the data
                X = data.drop(columns=["target_temp", "time"])
                y = data["target_temp"]
                # Train the model
                model.fit(X, y)
                model_filename = f"model_{hour}_{month}.pkl"
                joblib.dump(model, os.path.join(output_directory, model_filename))
                print(count)
                count += 1


# Directory containing your hour-month CSV files
data_directory = "E:\Learning Python\WeatherStuff\CSV_Splitter\CSV_Collection"
# Directory to save trained models
output_directory = "Trained_Models"

# make a random forest regressor with some parameters
gradient_booster = GradientBoostingRegressor(
    n_estimators=100, max_depth=5, learning_rate=0.1
)
# Directory containing your hour-month CSV files
data_directory = "E:\Learning Python\WeatherStuff\CSV_Splitter\CSV_Collection"
# Directory to save trained models
output_directory = "Trained_Models"

train_models(gradient_booster, data_directory, output_directory)
