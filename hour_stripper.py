import pandas as pd

df = pd.read_csv("new_csv.csv")

df["time"] = pd.to_datetime(df["time"])

# remove all columns where the minute is not 0
df = df[df["time"].dt.minute == 0]
df["temp"] = round(((df["temp"] * 9 / 5) + 32), 2)
df["hour"] = df["time"].dt.hour
df["month"] = df["time"].dt.month
print(df)

df.to_csv("processed.csv", index=False)
