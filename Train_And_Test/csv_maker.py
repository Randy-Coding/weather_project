# Create a CSV file with the provided data in two columns
import pandas as pd

# Data for the first column
first_column = [
    29.84,
    27.86,
    28.04,
    25.52,
    25.7,
    22.82,
    20.84,
    24.43,
    30.56,
    31.64,
    33.26,
    35.06,
    38.12,
    40.1,
    39.02,
    39.2,
    36.32,
    35.42,
    35.96,
    36.5,
    37.22,
    37.58,
    37.22,
    38.12,
]

# Data for the second column
second_column = [
    27.04,
    26.28,
    25.88,
    25.2,
    24.87,
    26.7,
    29.32,
    30.38,
    32.22,
    34.0,
    34.98,
    35.62,
    36.35,
    37.36,
    37.0,
    35.75,
    33.77,
    32.49,
    32.06,
    31.34,
    30.96,
    30.71,
    31.02,
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
