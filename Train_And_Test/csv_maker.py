# Create a CSV file with the provided data in two columns
import pandas as pd

# Data for the first column
first_column = [
    37.94,
    37.76,
    37.94,
    38.12,
    38.12,
    38.48,
    38.66,
    39.02,
    39.74,
    41.18,
    43.88,
    46.76,
    51.08,
    54.32,
    55.94,
    52.52,
    47.84,
    48.56,
    48.74,
    46.94,
    45.5,
    45.68,
    44.78,
    45.14,
]

# Data for the second column
second_column = [
    37.03,
    37.53,
    37.6,
    35.68,
    35.56,
    35.11,
    37.16,
    39.11,
    40.08,
    47.36,
    43.8,
    45.68,
    48.87,
    48.77,
    52.94,
    44.99,
    44.93,
    47.54,
    46.13,
    42.95,
    43.09,
    40.8,
    38.93,
    39.89,
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

import matplotlib.pyplot as plt

# Assuming df['Percent Change'] is already calculated and ready for plotting

plt.figure(figsize=(10, 6))

# Plotting the percent change
plt.plot(df["Percent Change"], marker="o", linestyle="-", color="blue")

# Setting title and labels
plt.title("Percent Change from First Column to Second Column")
plt.xlabel("Index")
plt.ylabel("Percent Change")

# Setting the X and Y axis scales
plt.xlim(0, len(df["Percent Change"]))  # For X-axis, from 0 to the length of your data
plt.ylim(-20, 70)  # For Y-axis, example range from -100% to 100%

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
