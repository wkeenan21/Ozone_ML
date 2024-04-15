
import os
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import math
import arcpy
arcpy.CheckOutExtension("Spatial")
arcpy.CheckOutExtension("GeoStats")
sr = arcpy.SpatialReference(4326)
arcpy.env.overwriteOutput = True
from datetime import timedelta

def generate_equally_spaced_points(min_lat, max_lat, distance):
    """
    Generate equally spaced points along a latitude given a distance between each point.

    Parameters:
    min_lat (float): The minimum latitude.
    max_lat (float): The maximum latitude.
    distance_meters (float): The distance in meters between each point.

    Returns:
    list: List of equally spaced points along the latitude.
    """
    num_points = int(math.ceil((max_lat - min_lat)/distance))
    # Generate equally spaced points
    points = []
    for i in range(num_points):
        lat = min_lat + i * distance
        points.append(lat)
    return points

def calcDist(latitude, distance_meters):
    # Earth's radius in meters
    earth_radius = 6371000  # Approximate value, can vary depending on the exact model of Earth used
    # Convert latitude to radians
    latitude_radians = math.radians(latitude)
    # Calculate the number of decimal degrees of latitude
    decimal_degrees_latitude = (distance_meters * 360) / (2 * math.pi * earth_radius * math.cos(latitude_radians))
    return decimal_degrees_latitude

def dt_to_name(dt):
    date = str(dt.date())
    time = str(dt.time())[0:2]
    return f'{date}_{time}'

import random

# THIS BLOCK CREATES A LIST OF LATITUDES AND LONGITUDES FOR EACH POINT
unawareFold = r'D:\Will_Git\Ozone_ML\Year2\Merged_Data\nh_unaware'
# Grab the coordinates for each site from here D:\Will_Git\Ozone_ML\Year2\Merged_Data\nh_unaware
horizons = [1,3,6,9,12,15,18,24]
random_list = random.sample(range(1,11), 10)
latDict = {}
lonDict = {}
weightDict = {}
for file in os.listdir(unawareFold):
    for horizon in horizons:
        site = file.replace('.csv', '')
        # if it's an unaware site
        if len(site) < 4:
            df = pd.read_csv(os.path.join(unawareFold, file))
            lat = df.reset_index()['point_latitude_x'].iloc[0]
            lon = df.reset_index()['point_longitude_x'].iloc[0]
            siteName = f'o3_sim_{site}'
            weightDict[siteName] = 1
        elif site == 'statics':
            continue
        # if its an aware site
        elif len(site) > 4:
            df = pd.read_csv(os.path.join(unawareFold, file))
            lat = df.reset_index()['latitude'].iloc[0]
            lon = df.reset_index()['longitude'].iloc[0]
            siteName = f'o3_sim_{site}_{horizon}'
            weightDict[siteName] = 5

        latDict[siteName] = lat
        lonDict[siteName] = lon

unAwareDfs = {}
awareDfs = {}

for horizon in horizons:
    # Merge 1 hour forecasts over stations with 1 hour unaware forecasts
    # data cleaning
    fold = r'D:\Will_Git\Ozone_ML\Year2\nh_results\forecast_csvs'
    awareDf = pd.read_csv(os.path.join(fold, f'forecasts_{horizon}.csv'))
    if horizon < 10:
        renames = {}
        for col in awareDf.columns:
            if 'o3' in col:
                renames[col] = col[0:-1]
        awareDf.rename(columns=renames, inplace=True)

    unAwareDf = pd.read_csv(os.path.join(fold, 'unaware_forecasts.csv'))
    awareDf['datetime'] = pd.to_datetime(awareDf['Unnamed: 0'])
    unAwareDf['datetime'] = pd.to_datetime(unAwareDf['Unnamed: 0'])
    awareDf.dropna(inplace=True)
    unAwareDf.dropna(inplace=True)

    # awareDf['datetime'] = awareDf['datetime'] + timedelta(hours=-7) # convert to mst
    # unAwareDf['datetime'] = awareDf['datetime'] + timedelta(hours=-7)

    awareDfs[horizon] = awareDf
    unAwareDfs[horizon] = unAwareDf


start_date = '2023-08-05 00:00:00'
end_date = '2023-08-05 01:00:00'
# Create the date range with 1-hour intervals
date_range = pd.date_range(start=start_date, end=end_date, freq='1H')

# this creates csvs for each hour for simulations
testHorizons = [1]
for horizon in testHorizons:
    # make a folder
    gdbFold = rf'D:\Will_Git\Ozone_ML\Year2\nh_results\ready_to_interp_csvs2\csvs\{horizon}Hour'
    if not os.path.exists(gdbFold):
        os.makedirs(gdbFold)
    print(f'horizon: {horizon}')
    errors = []
    for hour in date_range:
        awareDf = awareDfs[horizon]
        print(f'hour: {hour}')
        # get the hour in question
        # make copies of the aware data to give it higher weights
        thisHourA = awareDf[awareDf['datetime'] == hour]

        arrays = []
        WEIGHT = 1
        for i in range(WEIGHT):
            arrays.append(thisHourA)
        # combine them
        thisHourA = pd.concat(arrays, axis=1)
        # get unaware
        thisHourU = unAwareDf[unAwareDf['datetime'] == hour]

        # combine unaware with aware
        thisHour = pd.concat([thisHourA, thisHourU], axis=1)
        thisHour = thisHour.drop(columns=['Unnamed: 0', 'datetime']).reset_index(drop=True)

        thisHour = thisHour.transpose()
        thisHour.rename(columns={0:'o3'}, inplace=True)
        thisHour['site'] = thisHour.index
        thisHour['latitude'] = thisHour['site'].map(latDict)
        thisHour['longitude'] = thisHour['site'].map(lonDict)
        thisHour = thisHour.dropna()
        # add a random number to the end of each latitude for aware sites to make arcgis not ignore them
        thisHour['randoms'] = random.sample(range(1, 1000), len(thisHour))
        thisHour['randoms'] = thisHour['randoms'] / 10000000
        adjustedLats = thisHour['latitude'].where(thisHour['site'].str.len() < 13,
                                                  thisHour['latitude'] - thisHour['randoms'], axis=0)
        thisHour['latitude'] = adjustedLats

        if len(thisHour) < 151:
            print(f'not enough points for {hour}')

        # name things
        hour = hour + timedelta(hours=-7)
        hourString = str(hour).replace(' ', '_')
        hourString = hourString.replace('-', '_')
        day = hourString[0:10]
        hourString = str(hourString)[0:13]

        # here we say only interpolate aware
        thisHour = thisHourA

        # create a folder for csvs, create csv
        dayFolder = fr'D:\Will_Git\Ozone_ML\Year2\nh_results\ready_to_interp_csvs2\csvs\{horizon}Hour\{day}'
        if not os.path.exists(dayFolder):
            os.makedirs(dayFolder)
        outCsv = os.path.join(dayFolder, f'{hourString}.csv')
        #if not os.path.exists(outCsv):
        thisHour.to_csv(outCsv)

        # name the gdb
        gdbName = f'{horizon}Hour.gdb'
        if not arcpy.Exists(os.path.join(gdbFold, gdbName)):
            arcpy.management.CreateFileGDB(gdbFold, gdbName)
        fdName = f'fd_{day}'

        gdb = os.path.join(gdbFold, gdbName)
        featureDataset = os.path.join(gdb, fdName)
        if not arcpy.Exists(featureDataset):
            arcpy.management.CreateFeatureDataset(out_dataset_path=gdb, out_name=fdName, spatial_reference=sr)

        outFeature = os.path.join(featureDataset, f'fc_{hourString}')
        #if not arcpy.Exists(outFeature):
        arcpy.management.XYTableToPoint(in_table=outCsv, out_feature_class=outFeature, x_field='longitude', y_field='latitude',coordinate_system=sr)

        # # Set local variables
        # zField = "o3"
        # outRaster = os.path.join(dayFolder, f"{hourString}_w{WEIGHT}.tif")
        # outLayer = "throwAway"
        # # cell size in dec degrees
        # cellSize = calcDist(40, 1000) * 1
        # power = 0.5
        # # Execute IDW
        # out = arcpy.sa.Idw(in_point_features=outFeature, z_field=zField, cell_size=cellSize,power=power)
        # out.save(outRaster)
