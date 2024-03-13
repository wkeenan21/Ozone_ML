import pandas as pd
import numpy as np


# define base dir
baseDir = r"D:\Will_Git\Ozone_ML"
# Import data
O3J = pd.read_csv(fr"{baseDir}/Year2/Merged_Data/merge6.csv")
# do some preprocessing
# remove columns
remove = []
for col in O3J.columns:
    if 'Unnamed' in col or 'pressure' in col:
        remove.append(col)
O3J = O3J.drop(columns=remove)
# rename ozone
O3J.rename(columns={'sample_measurement':'o3'}, inplace=True)
# make columns for day and hour of day
O3J['datetime'] = pd.to_datetime(O3J['datetime'], utc=False)
# O3J.set_index('datetime', inplace=True)
# O3J.index = O3J['datetime'].tz_convert('America/Denver')

# remove values that are negative. They will be interpolated later
O3J['o3'].where(O3J['o3'] >= 0, other=np.nan, inplace=True)
# also remove values that are over 200 ppb. They will be interpolated later
O3J['o3'].where(O3J['o3'] < 0.2, other=np.nan, inplace=True)
#fifthP = O3J['o3'].quantile(q=0.05)

dfs = dict(tuple(O3J.groupby('site_name')))

site = 'Boulder'
view = dfs[site].corr()








