import joblib
from joblib import load
import pandas as pd
import numpy as np


def predict_future_temp(model, initial_data, steps):
    # Copy the initial data to avoid modifying the original DataFrame
    data = initial_data.copy()
    predicted_temp = model.predict(data.drop(columns=["target_temp", "time"]))
    return predicted_temp


# Load the model
model = load(
    r"C:\Users\Kage\Documents\GitHub\weather_project\Predicting\Random_Forest_Model.pkl"
)
new_data = pd.read_csv(
    r"C:\Users\Kage\Documents\GitHub\weather_project\Predicting\prediction_input_cleaned.csv"
)


# Make the prediction
predicted_temp_3hrs_ahead = predict_future_temp(model, new_data, 100)
print(f"Predicted temperature 3 hours ahead: {predicted_temp_3hrs_ahead} Fahrenheit")
