import matplotlib.pyplot as plt


def find_percentage_error(temperature, real_temp):
    error = abs(temperature - real_temp)
    percent_error = (error / real_temp) * 100
    return percent_error


def plot_percentage_error(temps, real_temp, model_names):
    plt.figure(figsize=(10, 6))
    errors = [find_percentage_error(temp, real_temp) for temp in temps]

    # Create a bar graph
    plt.bar(model_names, errors, color="skyblue")

    plt.ylabel("Percentage Error (%)")
    plt.title("Percentage Error for December 10th 11am Prediction")
    plt.xticks(rotation=45)  # Rotate model names for better readability
    plt.grid(True, axis="y")  # Enable grid only on the y-axis
    plt.tight_layout()  # Adjust plot to fit labels
    plt.show()


temps = [
    53.892011762603616,
    54.458443,
    55.184000000000005,
    53.598378355839316,
    54.506724773254014,
    53.43720958950148,
    54.50673207490576,
    53.825360684545885,
]

model_names = [
    "Catboost",
    "XGBoost",
    "RandomForest",
    "GradientBoosting",
    "LinearRegression",
    "LassoRegression",
    "RidgeRegression",
    "ElasticNet",
]

error = plot_percentage_error(temps, 53.06, model_names)

print(error)
