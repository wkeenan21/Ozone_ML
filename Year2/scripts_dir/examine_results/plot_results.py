import pandas as pd
import os
import datetime as dt
from Year2.scripts_dir.from_usgs.wills_functions import *

res = r"D:\Will_Git\Ozone_ML\Year2\results\southtable_5hour_24time_n.csv"

df = pd.read_csv(res)
df['date'] =pd.to_datetime(df['date'], utc=False)
filtered_df = df[df['date'] < '2023-07-15']
plotLines(filtered_df, 'date', yaxis='preds_4', yaxis2='actual_4')

