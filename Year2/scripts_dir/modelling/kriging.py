import os
import numpy as np
from matplotlib import pyplot as plt
from pyproj import Transformer
from pykrige.ok import OrdinaryKriging
import pandas as pd
import math
import rasterio
import rasterio.mask
from rasterio.plot import show
from rasterio.transform import Affine
from sklearn.metrics import mean_squared_error

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

def export_kde_raster(Z, XX, YY, min_x, max_x, min_y, max_y, proj, filename):
    '''Export and save a kernel density raster.'''
    # Get resolution
    xres = (max_x - min_x) / len(XX)
    yres = (max_y - min_y) / len(YY)
    # Set transform
    transform = Affine.translation(min_x - xres / 2, min_y - yres / 2) * Affine.scale(xres, yres)
    # Export array as raster
    with rasterio.open(
            filename,
            mode = "w",
            driver = "GTiff",
            height = Z.shape[0],
            width = Z.shape[1],
            count = 1,
            dtype = Z.dtype,
            crs = proj,
            transform = transform,
    ) as new_dataset:
            new_dataset.write(Z, 1)

def dt_to_name(dt):
    date = str(dt.date())
    time = str(dt.time())[0:2]
    return f'{date}_{time}'

results = r"D:\Will_Git\Ozone_ML\Year2\results\universal_one_hot\8_hour"

start_date = '2023-08-06 00:00:00'
end_date = '2023-08-06 23:00:00'
# Create the date range with 1-hour intervals
date_range = pd.date_range(start=start_date, end=end_date, freq='1H')

dfs = {}
sites = []
for file in os.listdir(results):
    path = os.path.join(results, file)
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'], utc=False)
    df['date'] = df['date'].dt.tz_localize(None)
    _ = file.find('_')
    site = file[0:_]
    print(site)
    # remove site word from column headers
    for col in df.columns:
        if site in col:
            newName = col.replace('_'+site, "")
            df.rename(columns={col:newName}, inplace=True)
    for col in df.columns:
        if 'preds' in col or 'actual' in col:
            df[col] = df[col]*1000
    sites.append(site)
    dfs[site] = df

hour_dfs = {}
for dt in date_range:
    print(dt)
    thisHour = []
    for df in dfs.values():
        site = df['site_name'].iloc[0]
        print(site)
        df = df[df['date'] == dt]
        thisHour.append(df)
    hours = pd.concat(thisHour)
    hour_dfs[str(dt)] = hours

# our bbox in 4326
bbox = [39.46,-105.63,40.17,-104.42]

# get the spacing for a grid
# approximately at 40 degrees latitude, 1000 meters between points
y_steps = generate_equally_spaced_points(bbox[0], bbox[2], calcDist(40, 1000))
x_steps = generate_equally_spaced_points(bbox[1], bbox[3], calcDist(40, 1000))

# interpolate the 1 hour predictions and actuals
for dt in date_range:
    df = hour_dfs[str(dt)].reset_index()
    # file name
    name = dt_to_name(dt)
    #
    # if len(df) != 11:
    #     raise Exception

    # Create ordinary kriging object for actual:
    AOK = OrdinaryKriging(x=df['lon'],y=df['lat'],z=df['actual'],variogram_model="linear",verbose=False,enable_plotting=False,coordinates_type="geographic")
    Z_pk_krig, sigma_squared_p_krig = AOK.execute("grid", x_steps, y_steps)
    if np.all(Z_pk_krig == Z_pk_krig[0]):
        print(f'{dt} actual is all the same')
        stop

    #export_kde_raster(Z=Z_pk_krig, XX=x_steps, YY=y_steps,min_x=min(x_steps), max_x=max(x_steps), min_y=min(y_steps), max_y=max(y_steps), proj=4326, filename=r"D:\Will_Git\Ozone_ML\Year2\results\interpolations\scratch\actual_{}.tif".format(name))
    # Do it for predictions
    preds = []
    for i in range(1):
        OK = OrdinaryKriging(df['lon'],df['lat'],df[f'preds_{i}'],variogram_model="linear",verbose=False,enable_plotting=False,coordinates_type="geographic")

        # Execute on grid:
        Z_pk_krig, sigma_squared_p_krig = OK.execute("grid", x_steps, y_steps)
        if np.all(Z_pk_krig == Z_pk_krig[0]):
            print(f'{dt} preds is all the same')
        #export_kde_raster(Z = Z_pk_krig, XX = x_steps, YY = y_steps,min_x = min(x_steps), max_x = max(x_steps), min_y = min(y_steps), max_y = max(y_steps),proj = 4326, filename = r"D:\Will_Git\Ozone_ML\Year2\results\interpolations\scratch\preds{}_{}.tif".format(i, name))

