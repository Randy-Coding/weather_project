# Create a CSV file with the provided data in two columns
import pandas as pd

# Data for the first column
first_column = [
    -1.7,
    -1.5,
    -1.7,
    -1.6,
    -1.7,
    -1.5,
    -0.4,
    99.4,
    250.5,
    473.3,
    605.9,
    711.0,
    778.4,
    745.7,
    677.9,
    512.4,
    331.5,
    148.7,
    5.6,
    -2.2,
    -2.0,
    -1.9,
    -2.1,
    -1.9,
]

# Data for the second column
second_column = [
    -1.7,
    -1.5,
    -1.7,
    -1.6,
    -1.7,
    -1.5,
    -0.4,
    99.4,
    250.5,
    473.3,
    605.9,
    711.0,
    778.4,
    740.9418871239143,
    667.4660443397331,
    533.1665635546062,
    370.3431519204944,
    213.35000893162737,
    75.2916321545415,
    -38.52961992035648,
    -103.33835944352018,
    -116.45640683979013,
    -89.40192715556145,
]

# Check if the lengths of columns are equal, if not, pad the shorter column with NaN
length_difference = len(first_column) - len(second_column)
if length_difference > 0:
    second_column += [float("nan")] * length_difference
elif length_difference < 0:
    first_column += [float("nan")] * abs(length_difference)
# find the greatest difference between the two columns
greatest_difference = 0
for value in range(0, len(first_column)):
    if abs(first_column[value] - second_column[value]) > greatest_difference:
        greatest_difference = abs(first_column[value] - second_column[value])


# Create a DataFrame with the data
df = pd.DataFrame({"First Column": first_column, "Second Column": second_column})

print(greatest_difference)

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
