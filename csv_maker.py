# Create a CSV file with the provided data in two columns
import pandas as pd

# Data for the first column
first_column = [
    35.96, 35.06, 33.44, 34.16, 33.98, 33.44, 32.9, 34.16, 37.22, 39.2, 40.64, 42.98, 44.6, 46.4, 48.56, 49.28, 49.46, 47.66, 44.6, 41.9, 41.72, 40.46, 38.66, 37.4
]


# Data for the second column
second_column = [
35.96, 35.06, 33.44, 34.16, 33.98, 33.44, 32.9, 34.16, 37.22, 39.2, 40.64, 42.98, 43.794075884979726, 45.44071041970215, 43.41403338910142, 44.54116109050521, 41.352040233418656, 43.075650992052225, 40.23784580646876, 39.675702052092454, 37.10372698486185, 37.05291107593261, 34.74742867122958, 33.745416419267460
]

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
