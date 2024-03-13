
import os
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import math
import arcpy
arcpy.CheckOutExtension("Spatial")
sr = arcpy.SpatialReference(4326)

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

results = r"D:\Will_Git\Ozone_ML\Year2\results\grid_results"

gridDf = r"D:\Will_Git\Ozone_ML\Year2\Merged_Data\10kgrid_merge.csv"
gridDf = pd.read_csv(gridDf)
gridDf['datetime'] = pd.to_datetime(gridDf['datetime'])
siteCoords = gridDf['mapped_o'].unique()
gridCoords = gridDf['coords_x'].unique()

# Create a dictionary
mapping_dict = {}
# create another dict to map coords to coords
osite_dict = {}
# Iterate over the DataFrame
# make grid df only August 12 because I know all the sites are represented that day
grid_sub = gridDf[gridDf['datetime'].dt.day_of_year == 224].reset_index()
for index, row in grid_sub.iterrows():
    column1_value = row['site_name']
    column2_value = row['coords_x']
    column3_value = row['coords_y']
    # Add key-value pair to the dictionary if the key doesn't exist
    if column2_value not in mapping_dict:
        mapping_dict[column2_value] = column1_value
        osite_dict[column1_value] = column3_value


start_date = '2023-05-10 23:00:00'
end_date = '2023-09-30 23:00:00'
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

    # for mapping o sites to hrrr points
    siteName = df['site_name'][0]
    df['o_site'] = mapping_dict[siteName]
    df['o_coords'] = osite_dict[mapping_dict[siteName]]

    # find the "site"
    site = file[:-4]
    # remove site word from column headers
    # for col in df.columns:
    #     if site in col:
    #         newName = col.replace('_'+site, "")
    #         df.rename(columns={col:newName}, inplace=True)

    # remove site word from column headers for coordinate sites
    for col in df.columns:
        if len(col) > 14:
            newName = col[0:7]
            df.rename(columns={col:newName}, inplace=True)

    for col in df.columns:
        if 'preds' in col or 'actual' in col:
            df[col] = df[col]*1000
    sites.append(site)
    dfs[site] = df

hour_dfs = {}
for dt in date_range:
    thisHour = []
    for df in dfs.values():
        site = df['site_name'].iloc[0]
        df = df[df['date'] == dt]
        thisHour.append(df)
    hours = pd.concat(thisHour)
    if len(hours['o_site'].unique()) > 10:
        hour_dfs[str(dt)] = hours
    else:
        print(f'skipping {dt}')

cell_size = calcDist(40, 1000)

# our bbox in 4326
bbox = [39.46,-105.63,40.17,-104.42]
arcpy.env.extent = arcpy.Extent(bbox[1], bbox[0], bbox[3], bbox[2])
arcpy.env.overwriteOutput = True
# sites
sites = ['Evergreen', 'Idaho Springs', 'Five Points', 'Welby', 'Highlands Ranch', 'Rocky Flats', 'Boulder', 'Chatfield Reservoir', 'Sunnyside', 'East Plains', 'South Table']
# prediction horizons
preds = []
for i in range(6):
    preds.append(f'preds_{i}')

# first we loop through each hour
# then through each site
# then through each horizon
baseDir = r'D:\Will_Git\Ozone_ML\Year2\results\interp_error_testing'

for key, df in hour_dfs.items():
    errors = []
    # we save data here
    hour = key[0:13].replace(' ', '_')
    hour = hour.replace('-', '_')
    name = 'h_'+hour
    # create feature dataset for each hour
    try:
        arcpy.management.CreateFeatureDataset(out_dataset_path=r'{}\ready_to_IDW.gdb'.format(baseDir),out_name=name, spatial_reference=sr)
    except:
        continue
    # create folder for rasters for each hour
    raster_dir1 = r'{}\{}'.format(baseDir, name+'_raster')
    if not os.path.exists(raster_dir1):
        os.makedirs(raster_dir1)
    # create folder for csvs for each hour
    csv_dir1 = r'{}\{}'.format(baseDir, name+'_csv')
    if not os.path.exists(csv_dir1):
        os.makedirs(csv_dir1)
    # do leave one out validation for each site
    for site in sites:
        # no spaces
        site_no_space = site.replace(' ', '_')
        # make folder for each site within each hour
        raster_dir2 = os.path.join(raster_dir1, site_no_space)
        if not os.path.exists(raster_dir2):
            os.makedirs(raster_dir2)

        out_csv = r'{}\no{}_{}.csv'.format(csv_dir1, site_no_space, name)
        outFeature = r'{}\ready_to_IDW.gdb\{}\no{}_hour_{}'.format(baseDir, name, site_no_space, name)

        df_one_out = df.where(df['o_site'] != site).dropna()
        if not os.path.exists(out_csv):
            df_one_out.to_csv(out_csv)

        if not arcpy.Exists(outFeature):
            arcpy.management.XYTableToPoint(in_table=out_csv, out_feature_class=outFeature, x_field='lon', y_field='lat', coordinate_system=sr)

        # prepare some data for the error calc
        # these lines grab the actual ozone value and it's coordinates
        just1 = df[df['o_site'] == site]
        actualO = just1['actual'].iloc[0]
        actual0_point = just1['o_coords'].iloc[0]
        # dict where we keep some data
        error = {'hour':hour, 'coord': actual0_point, 'site_name': site, 'actual': actualO}
        for pred in preds:
            # make name
            name2 = name + f'_{pred}'
            # save path
            save_path = r'{}\{}.tif'.format(raster_dir2, name2)
            if not os.path.exists(save_path):
                outRaster = arcpy.sa.Idw(in_point_features=outFeature, z_field=pred, cell_size=cell_size, power=0.5)
                outRaster.save(save_path)
                print(f'saved {name2}')

            # now go calc the error

            # these lines grab the value of the interpolated ozone from the raster
            pointy, pointx = actual0_point.replace(' ', '').split(',')
            pointx = float(pointx)
            pointy = float(pointy)
            point = [arcpy.Point(pointx, pointy)]
            ras_with_1value = arcpy.sa.ExtractByPoints(save_path, point, 'INSIDE')
            predictedO = ras_with_1value.mean
            error[pred] = predictedO
            errors.append(error)

    errorDf = pd.DataFrame.from_dict(errors)
    errorDf.to_csv(r'D:\Will_Git\Ozone_ML\Year2\results\interp_error_testing\result_csvs\{}.csv'.format(name))
    print(f'calculated error for {name}')



from sklearn import mean_squared_error

