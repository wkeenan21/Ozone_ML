import pandas as pd
import os

baseDir = r'D:\Will_Git'
preppedDir = fr'{baseDir}\Ozone_ML\Year2\Merged_Data\nh_unaware'

gridDf = pd.read_csv(r"D:\10k_grid_merges\10kgrid_merge4.csv")

# rename some stuff
gridDf.rename(columns={'coords_x':'point_id', 'site_name':'ozone_site', 'NLCD_mean':'NLCD'}, inplace=True)

remove = []
# drop some columns
for col in gridDf.columns:
    if 'Unnamed' in col or 'pressure' in col:
        remove.append(col)
gridDf = gridDf.drop(columns=remove)
# make columns for day and hour of day
gridDf['datetime'] = pd.to_datetime(gridDf['datetime'], utc=False)
gridDf['datetime'] = gridDf['datetime'].dt.tz_localize(None)
gridDf['date'] = gridDf['datetime']

# columns we want to interpolate
cols = ['t2m', 'r2', 'sp', 'dswrf', 'MAXUVV', 'MAXDVV', 'orog', 'u10', 'v10','pop_den'] #, 'no2', 'no2_bool'
count = 0
for ID in sorted(list(gridDf['point_id'].unique())):
    count += 1
    df = gridDf[gridDf['point_id'] == ID]
    df['site_name'] = count
    print(len(df))

    for col in cols:
        # interpolate everything. After this there should be no NANs that are timesize hours away from other NaNs
        df[col] = df[col].interpolate(limit=96) # limit=timesize

    df.to_csv(fr'{preppedDir}\{count}_id.csv')

static_list = []
for file in os.listdir(preppedDir):
    if 'static' not in file:
        df = pd.read_csv(os.path.join(preppedDir, file))
        static_vars = {}
        static_vars['site'] = df.reset_index()['site_name'][0]
        static_vars['orog'] = df.reset_index()['orog'][0]
        static_vars['pop_den'] = df.reset_index()['pop_den'][0]
        static_vars['NLCD'] = df.reset_index()['NLCD'][0]
        static_list.append(static_vars)

statics = pd.DataFrame.from_dict(static_list)
statics.to_csv(f"{preppedDir}\statics.csv")


