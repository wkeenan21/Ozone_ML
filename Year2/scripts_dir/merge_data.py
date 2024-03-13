import pandas as pd
import os
from functools import reduce
"""
List our file paths
"""
baseDir = r'D:\Will_Git\Ozone_ML'
windFold = fr"{baseDir}\Year2\HRRR_Data\training\wind"
tempFold = fr"{baseDir}\Year2\HRRR_Data\training\tempAndRH"
oFold = fr"{baseDir}\Year2\EPA_Data\v2"
mFold = fr"{baseDir}\Year2\HRRR_Data\training\meteorology"
nFold = fr"{baseDir}\Year2\EPA_Data\no2"

"""
First we merge everything into one df
"""

years = ['2021', '2022', '2023']
months = ['5', '6', '7', '8', '9']
thirties = ['06', '09']
thirty1s = ['05', '07', '08']
# dropping from wind and temp data
drop = ['step','heightAboveGround','latitude','longitude','valid_time','metpy_crs', 'gribfile_projection','y','x', 'point', 'time_point', 'Unnamed: 0']
# dropping from ozone data
drop2 = ['parameter_code', 'poc','datum','parameter', 'detection_limit', 'state_code', 'units_of_measure', 'units_of_measure_code','Unnamed: 0', 'sample_duration', 'sample_duration_code', 'sample_frequency', 'detection_limit', 'uncertainty','qualifier','method_type','method','method_code', 'state', 'date_of_last_change', 'cbsa_code']
# dropping from meteorology data
drop3 = ['surface', 'surface_x', 'surface_y']
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
        nPath = os.path.join(nFold, time2)
        mDf = pd.read_csv(mPath)
        wDf = pd.read_csv(wPath)
        tDf = pd.read_csv(tPath)
        oDf = pd.read_csv(oPath)
        nDf = pd.read_csv(nPath)
        wDf.drop(labels=drop, axis=1, inplace=True)
        tDf.drop(labels=drop, axis=1, inplace=True)
        mDf.drop(labels=drop3, axis=1, inplace=True)
        nDf.drop(labels=drop2, axis=1, inplace=True)
        if True:
            oDf.drop(labels=drop2, axis=1, inplace=True)
            mDf.rename(columns={'unknown_x': 'MAXUVV', 'unknown_y': 'MAXDVV'}, inplace=True)
            # make it datetime
            oDf['datetime'] = oDf['date_gmt'] + oDf['time_gmt']
            oDf['datetime'] = pd.to_datetime(oDf['datetime'], format='%Y-%m-%d%H:%M', utc=True)
            nDf['datetime'] = nDf['date_gmt'] + nDf['time_gmt']
            nDf['datetime'] = pd.to_datetime(nDf['datetime'], format='%Y-%m-%d%H:%M', utc=True)
            # a list of latitudes that share an o station and a no2 station
            collocatedN = [39.838119, 39.751184, 39.77949, 39.912799]

            nDf = nDf[nDf['latitude'].isin(collocatedN)]
            nDf.rename(columns={'sample_measurement': 'no2'}, inplace=True)
            oDf = pd.merge(oDf, nDf[['no2', 'datetime', 'latitude']], on=['datetime', 'latitude'], how='left')

            df_list = [wDf, tDf, mDf]
            hrrr = reduce(lambda left, right: pd.merge(left, right, on=['time', 'point_latitude', 'point_longitude'], how='outer'), df_list)
            #hrrr = pd.merge(wDf, tDf, on=['time', 'point_latitude', 'point_longitude'])
            hrrr.rename(columns={"point_latitude": "latitude", "point_longitude": "longitude", 'time':'datetime'}, inplace=True)
            hrrr['datetime'] = pd.to_datetime(hrrr['datetime'], format='%Y-%m-%d %H:%M:%S', utc=True)



            a = pd.merge(hrrr, oDf, on=['datetime', 'latitude', 'longitude'])
            dfs["{}-{}".format(month, year)] = a


a = pd.concat(dfs.values(), ignore_index=True)

# read in population data
pop_df = pd.read_csv(r"D:\Will_Git\Ozone_ML\Year2\BasicGIS\Pop_Den_O3.csv")
pop_df['site'] = pop_df['site_numbe'] + pop_df['county_cod']
pop_drop = []
for col in pop_df.columns:
    if 'RASTERVALU' not in col and col != 'site':
        pop_drop.append(col)
pop_df.drop(labels=pop_drop, axis=1, inplace=True)
pop_df.rename(columns={'RASTERVALU': 'pop_den'}, inplace=True)

# site numbers are not unique, only unique by county. So make a new column for unique sites
a['site'] = a['site_number'] + a['county_code']
# merge pop den with other data
a = a.merge(pop_df, on='site', how='inner')
# I wanna name the sites
a['site_name'] = 'placeholder'
a['site_name'].where(a['site'] != 50, other='Idaho Springs', inplace=True)
a['site_name'].where(a['site'] != 27, other='Boulder', inplace=True)
a['site_name'].where(a['site'] != 65, other='Rocky Flats', inplace=True)
a['site_name'].where(a['site'] != 70, other='South Table', inplace=True)
a['site_name'].where(a['site'] != 57, other='Sunnyside', inplace=True)
a['site_name'].where(a['site'] != 33, other='Five Points', inplace=True)
a['site_name'].where(a['site'] != 3002, other='Welby', inplace=True)
a['site_name'].where(a['site'] != 7, other='Highlands Ranch', inplace=True)
a['site_name'].where(a['site'] != 39, other='Chatfield Reservoir', inplace=True)
a['site_name'].where(a['site'] != 11, other='East Plains', inplace=True)
a['site_name'].where(a['site'] != 73, other='Evergreen', inplace=True)

# create an NO2 boolean column to designate what sites have it
no2_sites = ['Welby', 'Rocky Flats', 'Sunnyside', 'Five Points']
a['no2_bool'] = True
a['no2_bool'].where(a['site_name'].isin(no2_sites), other=False, inplace=True)

a.to_csv(fr"{baseDir}\Year2\Merged_Data\merge7.csv")


sites = a['latitude'].unique()
siteKey = {}
for i in range(len(sites)):
    siteKey[sites[i]] = str(i)

