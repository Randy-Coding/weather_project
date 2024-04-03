import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# This is taking in a file of weather data and giving it a database called "Weather"
weather = pd.read_csv(
    "Training_Input.csv",
)

# This is going to take all the missing data and will fill it in using a linear algorithm
weather["solar"] = weather["solar"].interpolate(method="linear")
weather["temp"] = weather["temp"].interpolate(method="linear")
weather["wind_spd"] = weather["wind_spd"].interpolate(method="linear")
weather["wind_dir"] = weather["wind_dir"].interpolate(method="linear")

"""
This is making the time column recognizable by pandas
and transferring from Celsius to freedom units
"""
weather["time"] = pd.to_datetime(weather["time"])
weather["temp"] = round((weather["temp"] * 9 / 5 + 32), 2)

"""
This is prepping the data for our machine learning algorithm.
It is taking past data from every 10 minutes, as well as creating a column called "target_temp".
target_temp is the temperature an hour ahead, and the machine learning model is going to use this
to predict future temperature.

It is also representing hour and month as the cyclic functions sine and cosine, so pandas knows that
hour 0 comes right after hour 23, ad well as letting pandas know that month 1 comes after month 12.
"""

step = 15
counter = 1
for i in range(15, 61, step):
    weather[f"temp_{i}min_ago"] = weather["temp"].shift(counter)
    counter += 1
weather["temp_rolling_mean"] = round((weather["temp"].rolling(window=60).mean()), 2)
weather["temp_rolling_std"] = round((weather["temp"].rolling(window=60).std()), 2)
weather["hour_sin"] = np.sin(2 * np.pi * weather["time"].dt.hour / 24)
weather["hour_cos"] = np.cos(2 * np.pi * weather["time"].dt.hour / 24)
weather["month_sin"] = np.sin(2 * np.pi * weather["time"].dt.month / 12)
weather["month_cos"] = np.cos(2 * np.pi * weather["time"].dt.month / 12)

weather["year"] = weather["time"].dt.year
weather["month"] = weather["time"].dt.month
weather["day"] = weather["time"].dt.day
weather["hour"] = weather["time"].dt.hour
weather["target_temp"] = weather.shift(-4)["temp"]

print(weather)

# Cleaning up the data by getting rid of all rows with NaN values
weather = weather.dropna()

# Split the data into training and testing data from before and after 2022
training_data = weather[weather["time"] < "2023-01-01"]
testing_data = weather[weather["time"] >= "2023-01-01"]


training_data.to_csv("Training_Data.csv", index=False)
testing_data.to_csv("Testing_Data.csv", index=False)
