import pandas as pd
data_df = pd.read_csv(r"C:\Users\princ\OneDrive\Pictures\Final Project of linear\Bike_resell_value_calc-main\Bike_resell_value_calc-main\LinearProjectModel\Used_Bikes.csv")

data_df.columns = [c.strip().lower() for c in data_df.columns]

BIKE_NAMES = sorted(data_df["bike_name"].dropna().unique().tolist())
CITIES = sorted(data_df["city"].dropna().unique().tolist())
OWNERS = sorted(data_df["owner"].dropna().unique().tolist())
BRANDS = sorted(data_df["brand"].dropna().unique().tolist())
