"""
Author: William Keenan
Date 9.26.2023
For pulling HRRR data from Herbie
"""

from herbie import FastHerbie
import pandas as pd
import os
baseDir = r"D:\Will_Git\Ozone_ML\Year2\HRRR_Data\fxx8\wind"
# Gets the EPA ozone sites
sites = pd.read_csv(r"D:\Will_Git\Ozone_ML\Year2\BasicGIS\10km_grid.csv")
toDrop = []
for col in sites.columns:
    if 'FID' != col and 'latitude' not in col and 'longitude' not in col:
        toDrop.append(col)
sites.drop(toDrop, axis=1, inplace=True) # drop some useless columns

missingDays = []
years = ['2023']
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
            print(f'day range: {days}')
            dayDict = {}
            for day in days:
                hours = pd.date_range(start=day, periods=24, freq="1H",)
                FH = FastHerbie(hours, model="hrrr", fxx=range(8))
                print('getting {}'.format(day))
                try:
                    #ds = FH.xarray(":(?:TMP|RH):2 m", remove_grib=False)
                    ds = FH.xarray(":[U|V]GRD:10 m", remove_grib=False)
                except:
                    try:
                        print('failed once, trying again')
                        ds = FH.xarray(":[U|V]GRD:10 m", remove_grib=False)
                        #ds = FH.xarray(":(?:TMP|RH):2 m", remove_grib=False)
                    except:
                        print(f'{day} no work')
                        missingDays.append(day)
                        continue
                points1 = ds.herbie.nearest_points(sites)
                master = points1.to_dataframe()
                master.reset_index(inplace=True)
                master.sort_values(by=['point', 'valid_time'], inplace=True)
                master.drop(labels=['step', 'latitude', 'longitude', 'valid_time', 'metpy_crs', 'gribfile_projection', 'y', 'x','point'], axis=1, inplace=True)
                print(f'got {day}')
                dayDict[str(day)] = master

                #pd.merge(wDf, tDf, on=['time', 'point_latitude', 'point_longitude'])
            monthDf = pd.concat(dayDict.values(), ignore_index=True)
            print(f'saving csv: {outPath}')
            monthDf.to_csv(outPath)
