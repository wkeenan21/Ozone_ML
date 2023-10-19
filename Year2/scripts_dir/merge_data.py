import pandas as pd
import os
from functools import reduce
"""
List our file paths
"""
windFold = r"D:\Will_Git\Ozone_ML\Year2\HRRR_Data\wind"
tempFold = r"D:\Will_Git\Ozone_ML\Year2\HRRR_Data\tempAndRH"
oFold = r"D:\Will_Git\Ozone_ML\Year2\EPA_Data"
mFold = r"D:\Will_Git\Ozone_ML\Year2\HRRR_Data\meteorology"

"""
First we merge everything into one df
"""

years = ['2021', '2022', '2023']
months = ['5', '6', '7', '8', '9']
thirties = ['06', '09']
thirty1s = ['05', '07', '08']
drop = ['step','heightAboveGround','latitude','longitude','valid_time','metpy_crs', 'gribfile_projection','y','x', 'point', 'time_point', 'Unnamed: 0']
drop2 = ['parameter_code', 'poc','datum','parameter', 'detection_limit', 'state_code', 'units_of_measure', 'units_of_measure_code','Unnamed: 0', 'sample_duration', 'sample_duration_code', 'sample_frequency', 'detection_limit', 'uncertainty','qualifier','method_type','method','method_code', 'state', 'date_of_last_change', 'cbsa_code']

dfs = {}

for year in years:
    for month in months:
        print(year, month)
        time = "year{}month{}.csv".format(year, month)
        time2 = "year{}month{}{}.csv".format(year, '0', month)
        wPath = os.path.join(windFold, time)
        tPath = os.path.join(tempFold, time)
        oPath = os.path.join(oFold, time2)
        mPath = os.path.join(mFold, time)
        mDf = pd.read_csv(mPath)
        wDf = pd.read_csv(wPath)
        tDf = pd.read_csv(tPath)
        oDf = pd.read_csv(oPath)
        wDf.drop(labels=drop, axis=1, inplace=True)
        tDf.drop(labels=drop, axis=1, inplace=True)
        if year != '2023' and (month != '8' or month != '9'):
            oDf.drop(labels=drop2, axis=1, inplace=True)
            mDf.rename(columns={'unknown_x': 'MAXUVV', 'unknown_y': 'MAXDVV'}, inplace=True)

            df_list = [wDf, tDf, mDf]
            hrrr = reduce(lambda left, right: pd.merge(left, right, on=['time', 'point_latitude', 'point_longitude'], how='outer'), df_list)
            #hrrr = pd.merge(wDf, tDf, on=['time', 'point_latitude', 'point_longitude'])
            hrrr.rename(columns={"point_latitude": "latitude", "point_longitude": "longitude", 'time':'datetime'}, inplace=True)
            hrrr['datetime'] = pd.to_datetime(hrrr['datetime'], format='%Y-%m-%d %H:%M:%S', utc=True)

            oDf['datetime'] = oDf['date_gmt'] + oDf['time_gmt']
            oDf['datetime'] = pd.to_datetime(oDf['datetime'], format='%Y-%m-%d%H:%M', utc=True)

            a = pd.merge(hrrr, oDf, on=['datetime', 'latitude', 'longitude'])
            dfs["{}-{}".format(month, year)] = a


a = pd.concat(dfs.values(), ignore_index=True)
a.to_csv(r"D:\Will_Git\Ozone_ML\Year2\Merged_Data\merge2.csv")


sites = a['latitude'].unique()
siteKey = {}
for i in range(len(sites)):
    siteKey[sites[i]] = str(i)

