import pandas as pd
import os
import matplotlib.pyplot as plt

df = pd.read_csv("D:\Will_Git\Ozone_ML\Year2\Merged_Data\merge8.csv")
df['sample_measurement'] = df['sample_measurement'] * 1000
rmses = [8.16,7.11,8.73,10.53,6.60,7.82,7.32,7.43,8.74,7.20,6.78]

# subset by site
stats = []
for i, site in enumerate(sites):
    statsDict = {}
    print(site)
    df_sub = df[df['site_name'] == site]
    print(df_sub['sample_measurement'].std())
    print(df_sub['sample_measurement'].mean())
    print(df_sub['sample_measurement'].median())

    statsDict['Site'] = site
    statsDict['Standard Deviation, Observed O3'] = df_sub['sample_measurement'].std()
    statsDict['Mean, Observed O3'] = df_sub['sample_measurement'].mean()
    #statsDict['Median'] = df_sub['sample_measurement'].median()
    statsDict['Model RMSE'] = rmses[i]
    stats.append(statsDict)

df = pd.DataFrame.from_dict(stats)

# Set the 'Site' column as the index for easier plotting
df.set_index('Site', inplace=True)

# Plotting
plt.figure(figsize=(10, 6))

# Get the number of sites
num_sites = len(df.index)

# Width of each bar
bar_width = 0.25

# Position of bars on x-axis
index = range(num_sites)

# Plot each statistic as a separate bar
for i, column in enumerate(df.columns):
    plt.bar([x + i * bar_width for x in index], df[column], width=bar_width, label=column)

plt.ylabel('Ozone (ppb)', fontsize=14)
plt.title('Statistics by Site', fontsize=14)
plt.xticks([x + bar_width for x in index], df.index, fontsize=11, rotation=45, ha='right')
plt.legend()
plt.tight_layout()

# Show plot
plt.show()



