# Create a CSV file with the provided data in two columns
import pandas as pd

# Data for the first column
first_column = [
    273.3,
    346.5,
    299.4,
    310.8,
    302.7,
    320.9,
    18.9,
    142.8,
    104.0,
    135.0,
    145.6,
    162.4,
    141.5,
    141.9,
    152.7,
    147.4,
    154.4,
    146.1,
    148.3,
    147.8,
    155.2,
    153.0,
]

# Data for the second column
second_column = [
    273.3,
    346.5,
    299.4,
    310.8,
    302.7,
    320.9,
    18.9,
    142.8,
    104.0,
    135.0,
    145.6,
    162.4,
    141.5,
    144.8518118708352,
    144.64276331163322,
    153.96041422933467,
    153.61775872162954,
    159.50178452091228,
    161.0510084260896,
    162.25604121400633,
    162.92044957813448,
    183.54371394435503,
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
