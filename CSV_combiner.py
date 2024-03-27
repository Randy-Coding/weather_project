import pandas as pd

# Load the CSV files into DataFrames (adjust the file paths as necessary)
df1 = pd.read_csv("data_1.csv")
df2 = pd.read_csv("data_2.csv")

# Concatenate the DataFrames
combined_df = pd.concat([df1, df2], ignore_index=True)

# remove duplicates
final_df = combined_df.drop_duplicates(subset=["TmStamp"], keep="first")

