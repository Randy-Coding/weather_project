# Create a CSV file with the provided data in two columns
import pandas as pd

# Data for the first column
first_column = [
    34.52,
    33.08,
    32.18,
    32.0,
    32.0,
    31.1,
    33.08,
    42.8,
    52.7,
    55.94,
    56.12,
    55.04,
    55.04,
    55.22,
    53.6,
    52.52,
    49.82,
    46.58,
    44.78,
    43.7,
    42.08,
    41.9,
]

# Data for the second column
second_column = [
    34.52,
    33.08,
    32.18,
    32.0,
    32.0,
    31.1,
    33.08,
    42.8,
    52.7,
    55.94,
    56.12,
    55.04,
    55.04,
    54.371161157975585,
    52.863183798091725,
    51.31199304889423,
    48.932228544437706,
    45.43288918867209,
    43.817894878638825,
    43.919514457841125,
    43.71376263235212,
    43.30167699197413,
]

# Check if the lengths of columns are equal, if not, pad the shorter column with NaN
length_difference = len(first_column) - len(second_column)
if length_difference > 0:
    second_column += [float("nan")] * length_difference
elif length_difference < 0:
    first_column += [float("nan")] * abs(length_difference)

# Create a DataFrame with the data
df = pd.DataFrame({"First Column": first_column, "Second Column": second_column})

print(df)

import matplotlib.pyplot as plt

df["Percent Change"] = (
    (df["Second Column"] - df["First Column"]) / df["First Column"] * 100
)

# Calculate the value difference between the two columns and store it in a new column
df["Value Difference"] = df["Second Column"] - df["First Column"]

# Assuming df['Value Difference'] is now calculated and ready for plotting

plt.figure(figsize=(10, 6))

# Plotting the value difference
plt.plot(df["Value Difference"], marker="o", linestyle="-", color="blue")

# Setting title and labels
plt.title("Value Difference from First Column to Second Column")
plt.xlabel("Index")
plt.ylabel("Value Difference")

# You might want to adjust the xlim and ylim based on the range of your data
plt.xlim(0, len(df["Value Difference"]))  # X-axis from 0 to the length of your data
# Adjust ylim according to the range of value differences you have
plt.ylim(df["Value Difference"].min() - 10, df["Value Difference"].max() + 10)

# Adding a grid for better readability
plt.grid(True)

# Display the plot
plt.show()
# Plot the first column as a line
plt.plot(df["First Column"], label="First Column")

# Plot the second column as a line
plt.plot(df["Second Column"], label="Second Column")

# Add labels and title to the plot
plt.xlabel("Index")
plt.ylabel("Value")
plt.title("Data Visualization")

# Add a legend to the plot
plt.legend()

# Display the plot
plt.show()
