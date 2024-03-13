import pandas as pd
import os
from functools import reduce
import math

def calculate_distance(x1, y1, x2, y2):
    """
    Calculate the distance between two points given their x and y coordinates.
    Parameters:
    x1 (float): x-coordinate of the first point.
    y1 (float): y-coordinate of the first point.
    x2 (float): x-coordinate of the second point.
    y2 (float): y-coordinate of the second point.
    Returns:
    float: The distance between the two points.
    """
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance

"""
List our file paths
"""
baseDir = r'D:\Will_Git\Ozone_ML'
windFold = fr"{baseDir}\Year2\HRRR_Data\testing\wind"
tempFold = fr"{baseDir}\Year2\HRRR_Data\testing\tempAndRH"
oFold = fr"{baseDir}\Year2\EPA_Data"
mFold = fr"{baseDir}\Year2\HRRR_Data\testing\meteorology"

"""
First we merge everything into one df
"""

years = ['2023']
months = ['5', '6', '7', '8', '9']
thirties = ['06', '09']
thirty1s = ['05', '07', '08']
# dropping from wind and temp data
drop = ['heightAboveGround', 'Unnamed: 0']
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
        mDf = pd.read_csv(mPath)
        wDf = pd.read_csv(wPath)
        tDf = pd.read_csv(tPath)
        oDf = pd.read_csv(oPath)
        wDf.drop(labels=drop, axis=1, inplace=True)
        tDf.drop(labels=drop, axis=1, inplace=True)
        mDf.drop(labels=drop3, axis=1, inplace=True)


        if True:
            oDf.drop(labels=drop2, axis=1, inplace=True)
            mDf.rename(columns={'unknown_x': 'MAXUVV', 'unknown_y': 'MAXDVV'}, inplace=True)

            df_list = [wDf, tDf, mDf]
            hrrr = reduce(lambda left, right: pd.merge(left, right, on=['time', 'point_latitude', 'point_longitude'], how='outer'), df_list)
            #hrrr = pd.merge(wDf, tDf, on=['time', 'point_latitude', 'point_longitude'])
            hrrr.rename(columns={'time':'datetime'}, inplace=True)
            hrrr['datetime'] = pd.to_datetime(hrrr['datetime'], format='%Y-%m-%d %H:%M:%S', utc=True)

            oDf['datetime'] = oDf['date_gmt'] + oDf['time_gmt']
            oDf['datetime'] = pd.to_datetime(oDf['datetime'], format='%Y-%m-%d%H:%M', utc=True)

            # create new columns in the hrrr df that map to the locations of the ozone stations
            hrrr['coords'] = hrrr['point_latitude'].astype(str) + ',' + hrrr['point_longitude'].astype(str)
            hrrr_coords = hrrr['coords'].unique()

            oDf['coords'] = oDf['latitude'].astype(str) + ',' + oDf['longitude'].astype(str)
            o_coords = oDf['coords'].unique()

            closest_os = {}
            for h_coord in hrrr_coords:
                distances = {}
                for o_coord in o_coords:
                    h = h_coord.split(',')
                    hy = float(h[0])
                    hx = float(h[1])
                    o = o_coord.split(',')
                    oy = float(o[0])
                    ox = float(o[1])
                    dist = calculate_distance(hx, hy, ox, oy)
                    distances[o_coord] = dist
                closest_o = min(distances, key=lambda k: distances[k])
                closest_os[h_coord] = closest_o

            hrrr['mapped_o'] = hrrr['coords'].map(closest_os)

            hrrr['latitude'], hrrr['longitude'] = hrrr['mapped_o'].str.split(',').str
            hrrr['latitude'] = hrrr['latitude'].astype(float)
            hrrr['longitude'] = hrrr['longitude'].astype(float)
            a = pd.merge(hrrr, oDf, on=['datetime', 'latitude', 'longitude'])
            dfs["{}-{}".format(month, year)] = a


a = pd.concat(dfs.values(), ignore_index=True)

# read in population data
pop_df = pd.read_csv(r"D:\Will_Git\Ozone_ML\Year2\BasicGIS\grid_extract.csv")
#pop_df['site'] = pop_df['site_numbe'] + pop_df['county_cod']

pop_df.rename(columns={'latitude':'point_latitude', 'longitude':'point_longitude', 'orog':'elev'}, inplace=True)

pop_drop = []
for col in pop_df.columns:
    if 'point' not in col and 'NLCD' not in col and 'elev' not in col:
        pop_drop.append(col)
pop_df.drop(labels=pop_drop, axis=1, inplace=True)

# make col for merging with a
pop_df['coords_x'] = pop_df['point_latitude'].astype(str) + ',' + pop_df['point_longitude'].astype(str)

# merge pop den with other data
a = a.merge(pop_df, on='coords_x', how='inner')

# site numbers are not unique, only unique by county. So make a new column for unique sites
a['site'] = a['site_number'] + a['county_code']

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
a.to_csv(fr"{baseDir}\Year2\Merged_Data\10kgrid_merge2.csv")


sites = a['latitude'].unique()
siteKey = {}
for i in range(len(sites)):
    siteKey[sites[i]] = str(i)

