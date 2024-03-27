import joblib
import pandas as pd
import os

import pandas as pd


def predict_temp(current_conditions, models_directory="Trained_Models"):
    """
    Predicts the temperature an hour into the future based on current conditions,
    using the appropriate model for the given date and time.

    Parameters:
    - current_conditions: A dictionary including 'date_time', 'temp', 'wind_dir', 'wind_spd', 'solar'.
    - models_directory: Path to the directory where the models are saved.

    Returns:
    - The predicted temperature an hour into the future.
    """
    # Extract date and time
    date_time = pd.to_datetime(current_conditions["time"])
    hour = date_time.hour
    month = date_time.month

    # Select and load the appropriate model
    model_filename = f"model_{hour}_{month}.pkl"
    model_path = os.path.join(models_directory, model_filename)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No model found for hour {hour} and month {month}.")

    model = joblib.load(model_path)

    # Prepare the input for prediction
    # Ensure you exclude 'date_time' from the input features
    input_features = {k: [v] for k, v in current_conditions.items() if k != "date_time"}
    input_df = pd.DataFrame(input_features)

    # Predict the future temperature
    predicted_temp = model.predict(input_df)[0]

    return predicted_temp


current_conditions = {
    "time": "2024-03-10 14:00:00",  # Example date and time
    "temp": 25,
    "wind_dir": 180,
    "wind_spd": 5,
    "solar": 300,
}


future_temp = predict_temp(current_conditions)
print(f"Predicted temperature an hour into the future: {future_temp}")
