# Create a CSV file with the provided data in two columns
import pandas as pd

# Data for the first column
first_column = [
37.76, 38.66, 38.3, 34.16, 33.08, 31.46, 31.64, 29.12, 29.3, 39.56, 43.7, 45.14, 44.24, 44.42, 44.6, 44.06, 42.8, 41.18, 40.1, 39.2, 37.94, 37.58, 36.5, 35.42]
# Data for the second column
second_column = [
37.76, 38.66, 38.3, 34.16, 33.08, 31.46, 31.64, 29.12, 29.3, 39.56, 43.7, 45.14, 44.24, 44.973215369005416, 45.988823981105725, 46.344878066914994, 45.73811852916778, 45.30443605129607, 44.54987275455097, 43.39362744511887, 42.224600129256494, 41.52730998926083, 39.96890431831922, 37.94408039508568]
# Check if the lengths of columns are equal, if not, pad the shorter column with NaN
length_difference = len(first_column) - len(second_column)
if length_difference > 0:
    second_column += [float("nan")] * length_difference
elif length_difference < 0:
    first_column += [float("nan")] * abs(length_difference)
# find the greatest difference between the two columns
greatest_difference = 0
for value in range(
    0, max(len(first_column), len(second_column))
):  # Added closing parenthesis
    if abs(first_column[value] - second_column[value]) > greatest_difference:
        greatest_difference = abs(first_column[value] - second_column[value])


# Create a DataFrame with the data
df = pd.DataFrame({"First Column": first_column, "Second Column": second_column})

print(greatest_difference)

import matplotlib.pyplot as plt

df["Percent Change"] = (
    (df["Second Column"] - df["First Column"]) / df["First Column"] * 100
)
plt.figure(figsize=(10, 6))
plt.plot(df["Percent Change"], marker="o", linestyle="-", color="blue")
plt.title("Percent Change from First Column to Second Column")
plt.xlabel("Index")
plt.ylabel("Percent Change")
plt.xlim(0, len(df["Percent Change"]))
plt.ylim(df["Percent Change"].min() - 10, df["Percent Change"].max() + 10)
plt.grid(True)
plt.show()
plt.plot(df["First Column"], label="Actual Values")
plt.plot(df["Second Column"], label="Predicted Values")
plt.xlabel("Index")
plt.ylabel("Value")
plt.title("Data Visualization")
plt.legend()
plt.show()
