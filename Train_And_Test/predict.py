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
    hour = current_conditions["hour"]
    month = current_conditions["month"]
    # Select and load the appropriate model
    model_filename = f"model_{hour}_{month}.pkl"
    model_path = os.path.join(models_directory, model_filename)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No model found for hour {hour} and month {month}.")

    model = joblib.load(model_path)

    input_df = pd.DataFrame([current_conditions])

    # Predict the future temperature
    predicted_temp = model.predict(input_df)[0]

    return predicted_temp


current_conditions = {
    "temp": 41.9,
    "wind_spd": 2.2,
    "wind_dir": 49.3,
    "solar": 342.7,
    "hour": 10,
    "month": 3,
}


future_temp = predict_temp(current_conditions)
print(f"Predicted temperature an hour into the future: {future_temp}")
