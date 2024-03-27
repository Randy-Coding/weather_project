import pandas as pd
import os

# Replace this with the path to your main dataset
input_csv_path = "Training_Input.csv"

# Replace this with the directory where you want to save the split CSV files
output_directory = "CSV_Collection"

# Ensure the output directory exists
os.makedirs(output_directory, exist_ok=True)

# Read the main dataset
data = pd.read_csv(input_csv_path)

# Convert 'time' to datetime if it's not already in that format
data["time"] = pd.to_datetime(data["time"])
# convert temp to farenheit
data["temp"] = round(((data["temp"] * 9 / 5) + 32), 2)
# Extract hour and month from 'time' to use for splitting
data["hour"] = data["time"].dt.hour
data["month"] = data["time"].dt.month
data["target_temp"] = data["temp"].shift(-4)
data = data.dropna()

# Loop through all hours and months and split the data accordingly
count = 0
for hour in range(24):
    for month in range(1, 13):
        # Filter the data for the current hour and month
        filtered_data = data[(data["hour"] == hour) & (data["month"] == month)]

        # Continue only if there's data for this hour and month
        if not filtered_data.empty:
            # Define the output file path
            output_file_path = os.path.join(output_directory, f"HH-{hour}-{month}.csv")

            # Save the filtered data to a CSV file
            filtered_data.to_csv(output_file_path, index=False)
            print(count)
            count += 1
