# Create a CSV file with the provided data in two columns
import pandas as pd

# Data for the first column
first_column = [
    462.9, 696.8, 743.1, 724.6, 649.1, 505.9, 330.6, 114.3, 5.1, -1.9, -1.8, -1.8, -1.8]

# Data for the second column
second_column = [
462.9, 696.8, 597.0821533203125, 546.0677490234375, 454.3128662109375, 437.3150329589844, 264.1799011230469, 62.420265197753906, -0.39554619789123535, -3.94468355178833, -4.737818717956543, -5.52502965927124, -4.131577014923096]

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
