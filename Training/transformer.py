import pandas as pd

df = pd.read_excel('AZ_data_cleaned.xlsx', sheet_name = "Electricity power training",  engine="openpyxl")
df['out.electricity.total.energy_consumption.kwh'] = pd.to_numeric(df['out.electricity.total.energy_consumption.kwh'], errors='ignore')