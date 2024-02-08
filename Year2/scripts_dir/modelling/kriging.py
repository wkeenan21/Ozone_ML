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

results = r"D:\Will_Git\Ozone_ML\Year2\results\universal_one_hot"

start_date = '2023-06-02 12:00:00'
end_date = '2023-06-02 23:00:00'
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
            print(col)
            newName = col.replace('_'+site, "")
            print(newName)
            df.rename(columns={col:newName}, inplace=True)

    dfs[site] = df
    sites.append(site)

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
bbox = [39.259770,-105.632996,40.172953,-104.237732]
# transform to 3857
# transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857")
# point1 = transformer.transform(bbox_4326[0],bbox_4326[1])
# point2 = transformer.transform(bbox_4326[2], bbox_4326[3])
# bbox = [point1[0], point1[1], point2[0], point2[1]]

# get the spacing for a grid
y_steps = generate_equally_spaced_points(bbox[0], bbox[2], calcDist(40, 1000))
x_steps = generate_equally_spaced_points(bbox[1], bbox[3], calcDist(40, 1000))

# interpolate the 'actual'
# Create ordinary kriging object:
OK = OrdinaryKriging(
    hour_dfs[str(dt)]['lon'],
    hour_dfs[str(dt)]['lat'],
    hour_dfs[str(dt)]['actual'],
    variogram_model="linear",
    verbose=False,
    enable_plotting=False,
    coordinates_type="geographic",
)

# Execute on grid:
Z_pk_krig, sigma_squared_p_krig = OK.execute("grid", x_steps, y_steps)

export_kde_raster(Z = Z_pk_krig, XX = x_steps, YY = y_steps,
                  min_x = min(x_steps), max_x = max(x_steps), min_y = min(y_steps), max_y = max(y_steps),
                  proj = 4326, filename = r"D:\Will_Git\Ozone_ML\Year2\results\interpolations\scratch\actual{}.tif".format(str(dt)[0:10]))

# interpolate the predictions
for i in range(6):
# Create ordinary kriging object:
    OK = OrdinaryKriging(
        hour_dfs[str(dt)]['lon'],
        hour_dfs[str(dt)]['lat'],
        hour_dfs[str(dt)][f'preds_{i}'],
        variogram_model="linear",
        verbose=False,
        enable_plotting=False,
        coordinates_type="geographic",
    )

    # Execute on grid:
    Z_pk_krig, sigma_squared_p_krig = OK.execute("grid", x_steps, y_steps)

    export_kde_raster(Z = Z_pk_krig, XX = x_steps, YY = y_steps,
                      min_x = min(x_steps), max_x = max(x_steps), min_y = min(y_steps), max_y = max(y_steps),
                      proj = 4326, filename = r"D:\Will_Git\Ozone_ML\Year2\results\interpolations\scratch\pred{}_{}.tif".format(i, str(dt)[0:10]))