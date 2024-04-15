import pandas as pd
import os
from sklearn.metrics import mean_squared_error

fold = r'D:\Will_Git\Ozone_ML\Year2\nh_results\ready_to_interp_csvs\v2\error_csvs'
horizons = [1,3,6,9,12,15,18,24]
WEIGHT = 2
for h in horizons:
    print(f'horizon {h}')
    df = pd.read_csv(os.path.join(fold, f'{h}h_{WEIGHT}w_error.csv'))
    avermse = mean_squared_error(df['actual'], df['prediction'], squared=False)
    print(f'ave rmse: {avermse}')
    for site in list(df['missing_site'].unique()):
        print(site)
        df2 = df[df['missing_site'] == site]
        rmse = mean_squared_error(df2['actual'], df2['prediction'], squared=False)
        print(rmse)

