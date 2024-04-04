# Create a CSV file with the provided data in two columns
import pandas as pd

# Data for the first column
first_column = [
    1.0,
    0.7,
    0.7,
    0.9,
    0.9,
    1.0,
    1.2,
    1.7,
    1.8,
    2.6,
    3.3,
    2.3,
    2.7,
    2.7,
    1.6,
    2.4,
    1.6,
    1.8,
    1.6,
    1.3,
    1.2,
    1.5,
    1.8,
    1.9,
]


# Data for the second column
second_column = [
    1.0,
    0.7,
    0.7,
    0.9,
    0.9,
    1.0,
    1.2,
    1.7,
    1.8,
    2.6,
    3.3,
    2.3,
    2.458252349611265,
    2.53480364582277,
    2.5099693167098045,
    2.371540842913917,
    2.2593233649342253,
    2.0638378697562505,
    1.5922466252554708,
    1.1464352922497414,
    1.0126785434562429,
    1.0342512726286848,
    1.0722630708488161,
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
