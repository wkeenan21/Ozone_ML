from herbie import Herbie
from herbie import FastHerbie
import requests
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# Gets the EPA ozone sites
sites = pd.read_csv(r"D:\Will_Git\DU-Thesis\Year2\BasicGIS\ozone_sites.csv")
toDrop = []
for col in sites.columns:
    if 'FID' != col and 'latitude' not in col and 'longitude' not in col:
        toDrop.append(col)
sites.drop(toDrop, axis=1, inplace=True) # drop some useless columns


start = '2021-05-01'
days = pd.date_range(start=start, periods=30, freq="1D")
dayDict = {}
for day in days:
    hours = pd.date_range(start=day, periods=24, freq="1H",)
    FH = FastHerbie(hours, model="hrrr")
    print('getting {}'.format(day))
    ds = FH.xarray(":(?:TMP|RH):2 m", remove_grib=True)
    points = ds.herbie.nearest_points(sites)
    df = points.to_dataframe()
    df.reset_index(inplace=True)
    df.sort_values(by=['point', 'valid_time'], inplace=True)
    dayDict[day] = df

ag = {}
for day in days:
    daystr = str(day)[0:10]
    ag[daystr] = dayDict[day]
    ag[daystr]['time_point'] = ag[daystr]['time'].astype(str) + ' ' + ag[daystr]['point'].astype(str)
    #ag[daystr].set_index('time_point')

mayTempAndRH = pd.concat(ag.values(), ignore_index=True)
mayTempAndRH.to_csv(r'D:\Will_Git\DU-Thesis\Year2\HRRR_Data\tempAndRH\may.csv')

ex = ag['2021-05-01']