from Testing.predict import predict_master

print(
    "Welcome to my weather prediction project. Project automatically has weather data loaded from the BNL website. You can replace it under the 'Prediction_Data' directory."
)

input = input(
    "Please enter the weather you would like predicted. Valid inputs are 'temp', 'wind_spd', 'wind_dir', and 'solar'."
)

result = predict_master()

real_values = result[1]
predicted_values = result[2]

print(f"The {input} values for that day are: {real_values[input]}")
print(f"The predicted {input} values for that day are: {predicted_values[input]}")
