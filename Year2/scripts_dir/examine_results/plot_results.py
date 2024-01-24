import pandas as pd
import os
from Year2.scripts_dir.from_usgs.wills_functions import *

res = r"D:\Will_Git\Ozone_ML\Year2\results\southtable_5hour_24time_n.csv"

df = pd.read_csv(res)

plotLines(df, 'date', yaxis='preds_0', yaxis2='actual_0')

