import pandas as pd
import os
from sklearn.metrics import mean_squared_error

horizons = [1,3,6,9,12,15,18,24]
horizons = [6]
sites = ['Evergreen', 'Idaho Springs', 'Five Points', 'Welby', 'Highlands Ranch', 'Rocky Flats', 'Boulder', 'Chatfield Reservoir', 'Sunnyside', 'East Plains', 'South Table']
for h in horizons:
    print(h)
    df = pd.read_csv(rf'D:\Will_Git\Ozone_ML\Year2\nh_results\forecast_csvs\forecasts_{h}.csv', parse_dates=['Unnamed: 0']).dropna()
    df = df.rename(columns={'Unnamed: 0': 'datetime'})
    # subset the df for high ozone

    for site in sites:
        print(site)

        obs = 'obs'
        sim = 'sim'
        res = 'res'
        colObs = f'o3_{obs}_{site}_{h}_'
        colSim = f'o3_{sim}_{site}_{h}_'
        colRes = f'o3_{res}_{site}_{h}'
        dfHigh = df[df[colObs] > 65]

        months = [5,6,7,8,9]
        for m in months:
            dfMonth = df[df['datetime'].dt.month == m]
            monthRmse = mean_squared_error(dfMonth[colObs], dfMonth[colSim], squared=False)
            print(m, monthRmse)

        df[colRes] = df[colObs] - df[colSim]

        highRmse = mean_squared_error(dfHigh[colObs], dfHigh[colSim], squared=False)
        Rmse = mean_squared_error(df[colObs], df[colSim], squared=False)
        print(f'high rmse: {highRmse}, totalRmse: {Rmse}')

import os
import matplotlib
try:
    matplotlib.use('TkAgg')
except:
    print('no TkAgg')
import matplotlib.pyplot as plt

