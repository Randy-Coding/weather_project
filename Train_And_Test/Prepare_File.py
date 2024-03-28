import numpy as np
import pandas as pd
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib


def train_rnn_models(data_directory, output_directory):
    for hour in range(24):
        for month in range(1, 13):
            file_path = os.path.join(data_directory, f"HH-{hour}-{month}.csv")
            if os.path.exists(file_path):
                data = pd.read_csv(file_path)
                data.sort_values("time", inplace=True)

                # Assuming 'temp', 'wind_spd', 'wind_dir', and 'solar' are the features
                X = data[["temp", "wind_spd", "wind_dir", "solar"]].values
                # Target variables
                y = data[
                    [
                        "target_temp",
                        "target_wind_spd",
                        "target_wind_dir",
                        "target_solar",
                    ]
                ].values

                # Standardize features
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                # Split the dataset
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y, test_size=0.2, random_state=42, shuffle=False
                )

                # Reshape input to be [samples, time steps, features] which is required for RNN
                X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
                X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

                # Define the RNN model
                model = Sequential()
                model.add(
                    SimpleRNN(
                        50,
                        activation="relu",
                        input_shape=(X_train.shape[1], X_train.shape[2]),
                    )
                )
                model.add(Dense(4))  # Predicting 4 targets

                # Compile the model
                model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")

                # Fit the model
                model.fit(
                    X_train,
                    y_train,
                    epochs=100,
                    batch_size=32,
                    validation_data=(X_test, y_test),
                    verbose=2,
                )

                # Save the model
                model_filename = os.path.join(
                    output_directory, f"rnn_model_{hour}_{month}.keras"
                )
                model.save(model_filename)


data_directory = "CSV_Collection"
output_directory = "Trained_Models"
# Ensure the output directory exists
os.makedirs(output_directory, exist_ok=True)

# Train RNN models
train_rnn_models(data_directory, "Trained_Models")
