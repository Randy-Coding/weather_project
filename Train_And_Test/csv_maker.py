# Create a CSV file with the provided data in two columns
import pandas as pd

# Data for the first column
first_column = [
    56.3,
    57.56,
    57.56,
    54.32,
    54.86,
    49.64,
    48.92,
    48.92,
    52.16,
    47.12,
]

# Data for the second column
second_column = [34.72, 39.95, 42.29, 42.05, 45.63, 44.16, 45.67, 48.04, 50.04, 48.89]

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
