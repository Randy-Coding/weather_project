import joblib
import pandas as pd
import os

import pandas as pd


def predict_temp(
    current_conditions, hours_into_future=1, models_directory="Trained_Models"
):
    """
    Predicts the temperature for a few hours into the future based on current conditions,
    using the appropriate model for the given hour and month.

    Parameters:
    - current_conditions: A dictionary including 'temp', 'wind_dir', 'wind_spd', 'solar', 'hour', and 'month'.
    - hours_into_future: The number of hours into the future to predict.
    - models_directory: The directory where the models are saved.

    Returns:
    - A list of predicted temperatures for each hour into the future.
    """
    predicted_temps = []
    hour = current_conditions["hour"]
    month = current_conditions["month"]

    for _ in range(hours_into_future):
        # Determine the filename for the model corresponding to the current hour and month
        model_filename = f"model_{hour}_{month}.pkl"
        model_path = os.path.join(models_directory, model_filename)

        # Check if the model exists; if not, raise an error
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"No model found for hour {hour} and month {month}."
            )

        model = joblib.load(model_path)

        input_df = pd.DataFrame([current_conditions])

        # Perform the prediction
        forecast = model.predict(input_df)[0]
        predicted_temp = forecast[0]
        predicted_temps.append(round((predicted_temp), 2))
        current_conditions["temp", "wind_spd", "wind_dir", "solar"] = forecast

        # Update the hour for the next prediction, rolling over to the next day if needed
        hour = (hour + 1) % 24

    return predicted_temps


current_conditions = {
    "temp": 27.86,
    "wind_spd": 1.1,
    "wind_dir": 302.6,
    "solar": -1.9,
    "hour": 1,
    "month": 3,
}
hours_into_future = 23
future_temp = predict_temp(current_conditions, hours_into_future)
print(
    f"Predicted temperature {hours_into_future} hours into the future:\n {future_temp}"
)
