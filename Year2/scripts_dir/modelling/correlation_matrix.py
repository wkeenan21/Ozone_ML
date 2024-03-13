import pandas as pd

df = pd.read_csv(r"D:\Will_Git\Ozone_ML\Year2\Merged_Data\merge5.csv")
df = df.dropna()
df.rename(columns={'sample_measurement':'O3', 'r2':'Relative Humidity', 't2m': 'Temperature', 'orog':'Elevation', 'dswrf': 'Downward Short-Wave', 'MAXUVV':'Hourly Max Upward', 'MAXDVV': 'Hourly Max Downward', 'u10':'U Component of Wind', 'v10': 'V Component of Wind', 'RASTERVALU':'Population Density'}, inplace=True)

import matplotlib.pyplot as plt

df2 = df[['O3','Relative Humidity','Temperature','Elevation','Downward Short-Wave','Hourly Max Upward','Hourly Max Downward','U Component of Wind','V Component of Wind', 'sp']]

import seaborn as sns

f, ax = plt.subplots(figsize=(10, 8))
corr = df2.corr()
sns.heatmap(corr,
    cmap=sns.diverging_palette(220, 10, as_cmap=True),
    vmin=-1.0, vmax=1.0,
    square=True, ax=ax)