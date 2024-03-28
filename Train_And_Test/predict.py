import pandas as pd
import os
import tensorflow as tf  # Import TensorFlow
from tensorflow.keras.models import load_model
from keras.models import load_model
from keras.losses import MeanSquaredError
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError


def predict_temp_nn(
    current_conditions, hours_into_future=1, models_directory="Trained_Models"
):
    """
    Predicts the temperature for a few hours into the future based on current conditions,
    using the appropriate neural network model for the given hour and month.

    Parameters:
    - current_conditions: A dictionary including 'temp', 'wind_dir', 'wind_spd', 'solar', 'hour', and 'month'.
    - hours_into_future: The number of hours into the future to predict.
    - models_directory: The directory where the neural network models are saved.

    Returns:
    - A list of predicted temperatures for each hour into the future.
    """
    predicted_temps = []
    hour = current_conditions["hour"]
    month = current_conditions["month"]

    for _ in range(hours_into_future):
        # Determine the filename for the model corresponding to the current hour and month
        model_filename = (
            f"rnn_model_{hour}_{month}.h5"  # Adjusted for TensorFlow SavedModel format
        )
        model_path = os.path.join(models_directory, model_filename)

        # Check if the model exists; if not, raise an error
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"No model found for hour {hour} and month {month}."
            )

        # Load the TensorFlow/Keras model

        model = load_model(model_path)
        # Prepare the input data in the shape expected by the model
        # Assuming the model expects a numpy array of shape [samples, time steps, features]
        input_features = pd.DataFrame(
            [current_conditions], columns=["temp", "wind_spd", "wind_dir", "solar"]
        )
        input_features = input_features.values.reshape(
            (1, 1, input_features.shape[1])
        )  # Reshape for the RNN

        # Perform the prediction
        forecast = model.predict(input_features)[0]
        predicted_temp = forecast[0]
        predicted_temps.append(round(predicted_temp, 2))

        # Assuming the model outputs predictions for "temp", "wind_spd", "wind_dir", "solar" in this exact order
        current_conditions["temp"] = forecast[0]
        current_conditions["wind_spd"] = forecast[1]
        current_conditions["wind_dir"] = forecast[2]
        current_conditions["solar"] = forecast[3]

        # Update the hour for the next prediction, rolling over to the next day if needed
        hour = (hour + 1) % 24

    return predicted_temps


# Example usage
current_conditions = {
    "temp": 29.84,
    "wind_spd": 1.6,
    "wind_dir": 316,
    "solar": -1.9,
    "hour": 0,
    "month": 3,
}
hours_into_future = 24
models_directory = "Trained_Models"
future_temp = predict_temp_nn(current_conditions, hours_into_future, models_directory)
print(
    f"Predicted temperature {hours_into_future} hours into the future:\n {future_temp}"
)
