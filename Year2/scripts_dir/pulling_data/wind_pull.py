"""
Author: William Keenan
Date 9.26.2023
For pulling HRRR data from Herbie
"""

from herbie import FastHerbie
import pandas as pd
import os
baseDir = r"D:\Will_Git\Ozone_ML\Year2\HRRR_Data\testing\wind"
# Gets the EPA ozone sites
sites = pd.read_csv(r"/Year2/BasicGIS/10km_grid.csv")
toDrop = []
for col in sites.columns:
    if 'FID' != col and 'latitude' not in col and 'longitude' not in col:
        toDrop.append(col)
sites.drop(toDrop, axis=1, inplace=True) # drop some useless columns

missingDays = []
years = ['2021', '2022', '2023']
for year in years:
    print(year)
    start = '{}-04-30'.format(year)
    months = pd.date_range(start=start, periods=5, freq="1M")
    for month in months:
        month = month + pd.Timedelta(days=1)
        outPath = r"{}\year{}month{}.csv".format(baseDir, year, month.month)
        if not os.path.exists(outPath):
            print(outPath)
            thirties = [6, 9]
            thirty1s = [5, 7, 8]
            print('month: {}'.format(month))
            if month.month in thirties:
                periods = 30
            elif month.month in thirty1s:
                periods = 31
            days = pd.date_range(start=month, periods=periods, freq="1D")
            dayDict = {}
            for day in days:
                try:
                    print(day)
                    hours = pd.date_range(start=day, periods=24, freq="1H",)
                    FH = FastHerbie(hours, model="hrrr", fxx=range(0,6), save_dir=r"/hrrr")
                    print('getting {}'.format(day))
                    #ds = FH.xarray(":(?:TMP|RH):2 m", remove_grib=True)
                    ds = FH.xarray(":[U|V]GRD:10 m", remove_grib=False)
                    points1 = ds.herbie.nearest_points(sites)
                    master = points1.to_dataframe()
                    master.reset_index(inplace=True)
                    master.sort_values(by=['point', 'valid_time'], inplace=True)
                    master.drop(labels=['latitude', 'longitude', 'metpy_crs', 'gribfile_projection', 'y', 'x','point'], axis=1, inplace=True)
                    dayDict[str(day)] = master
                except:
                    missingDays.append(day)

                #pd.merge(wDf, tDf, on=['time', 'point_latitude', 'point_longitude'])
            monthDf = pd.concat(dayDict.values(), ignore_index=True)
            monthDf.to_csv(r"{}\year{}month{}.csv".format(baseDir, year, month.month))
            print('missing days in {}: {}'.format(month.month, missingDays))

