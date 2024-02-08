"""
Author: William Keenan
Date 9.26.2023
For pulling HRRR data from Herbie
"""

from herbie import FastHerbie
import pandas as pd
import os
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
        outPath = r"D:\Will_Git\Ozone_ML\Year2\HRRR_Data\testing\meteorology\year{}month{}.csv".format(year, month.month)
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
                    FH = FastHerbie(hours, model="hrrr")
                    print('getting {}'.format(day))
                    #ds = FH.xarray(":(?:TMP|RH):2 m", remove_grib=True)
                    #ds = FH.xarray(":[U|V]GRD:10 m", remove_grib=True)
                    varDF = {}
                    variables = [':MAXUVV:', ':MAXDVV:', ':DSWRF:', ':HGT:surface', ':PRES:surface']
                    for var in variables:
                        ds = FH.xarray(var, remove_grib=True)
                        points1 = ds.herbie.nearest_points(sites)
                        varDF[var] = points1.to_dataframe()
                        varDF[var].reset_index(inplace=True)
                        varDF[var].sort_values(by=['point', 'valid_time'], inplace=True)
                        varDF[var].drop(labels=['step', 'latitude', 'longitude', 'valid_time', 'metpy_crs', 'gribfile_projection', 'y', 'x','point'], axis=1, inplace=True)
                        if var == ':MAXUVV:':
                            varDF[var].rename(columns={'unknown':'MAXUVV'})
                        elif var == ':MAXDVV:':
                            varDF[var].rename(columns={'unknown': 'MAXDVV'})

                    variables = [':MAXDVV:', ':DSWRF:', ':HGT:surface', ':PRES:surface']
                    master = varDF[':MAXUVV:']
                    for var in variables:
                        master = pd.merge(varDF[var], master, on=['time', 'point_latitude', 'point_longitude'])

                    dayDict[str(day)] = master
                except Exception as error:
                    print(error)
                    date = "{}_{}_{}".format(day, month.month, year)
                    print(date)
                    missingDays.append(date)

                #pd.merge(wDf, tDf, on=['time', 'point_latitude', 'point_longitude'])
            monthDf = pd.concat(dayDict.values(), ignore_index=True)
            monthDf.to_csv(r"D:\Will_Git\Ozone_ML\Year2\HRRR_Data\testing\meteorology\year{}month{}.csv".format(year, month.month))

#
# ag = {}
# for day in days:
#     daystr = str(day)[0:10]
#     ag[daystr] = dayDict[day]
#     ag[daystr]['time_point'] = ag[daystr]['time'].astype(str) + ' ' + ag[daystr]['point'].astype(str)
#     #ag[daystr].set_index('time_point')

# start = '2021-05-31'
# days = pd.date_range(start=start, periods=62, freq="1D")
# dayDict = {}
# for day in days:
#     hours = pd.date_range(start=day, periods=24, freq="1H",)
#     FH = FastHerbie(hours, model="hrrr")
#     print('getting {}'.format(day))
#     #ds = FH.xarray(":(?:TMP|RH):2 m", remove_grib=True)
#     ds = FH.xarray(":[U|V]GRD:10 m", remove_grib=True)
#     points = ds.herbie.nearest_points(sites)
#     df = points.to_dataframe()
#     df.reset_index(inplace=True)
#     df.sort_values(by=['point', 'valid_time'], inplace=True)
#     dayDict[day] = df
#
# mayTempAndRH = pd.concat(ag.values(), ignore_index=True)
# print('sending to csv')
# mayTempAndRH.to_csv(r"D:\Will_Git\Ozone_ML\Year2\HRRR_Data\wind\year{}month{}.csv".format(year,month.month))
#
#
# mayTempAndRH = pd.concat(ag.values(), ignore_index=True)
# mayTempAndRH.to_csv(r"C:\Users\willy\Documents\GitHub\Ozone_ML\Year2\HRRR_Data\wind\june.csv")

