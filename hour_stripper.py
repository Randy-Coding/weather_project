import pandas as pd

df = pd.read_csv("historical_data.csv")

df.rename(
    columns={
        "TmStamp": "time",
        "temp_2": "temp",
        "speed_10": "wind_spd",
        "dir_10": "wind_dir",
    },
    inplace=True,
)
df.drop(["temp_10"], axis=1, inplace=True)
# remove all rows where the minute is not 0
df["time"] = pd.to_datetime(df["time"])
df = df[df["time"].dt.minute == 0]
df["temp"] = round(((df["temp"] * 9 / 5) + 32), 2)
df["hour"] = df["time"].dt.hour
df["month"] = df["time"].dt.month
for temp in df["temp"]:
    print(temp, ",")

df.to_csv("processed.csv", index=False)
