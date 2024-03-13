import datetime as dt
from datetime import timedelta
import os
from datetime import date
import pandas as pd
import numpy as np
import math
from math import log10, floor, log
import scipy.stats
import sklearn.preprocessing as sk
#import seaborn as sn
from sklearn.metrics import mean_squared_error
import sys

def add_time_columns(df, datetime_column_name):
    """
    Add 'day_of_year' and 'hour_of_day' columns to a DataFrame based on a datetime column.

    Parameters:
    - df: pandas DataFrame
    - datetime_column_name: str, the name of the datetime column in the DataFrame

    Returns:
    - DataFrame with additional 'day_of_year' and 'hour_of_day' columns
    """
    # Convert the datetime column to datetime type if it's not already
    df[datetime_column_name] = pd.to_datetime(df[datetime_column_name])

    # Add 'day_of_year' and 'hour_of_day' columns
    df['day_of_year'] = df[datetime_column_name].dt.dayofyear
    df['hour_of_day'] = df[datetime_column_name].dt.hour

    return df

def fill_missing_hours(df, datetime_column_name, target_months, constant_columns):
    """
    Fill missing hours in a DataFrame by adding rows for every hour in the range.

    Parameters:
    - df: pandas DataFrame
    - datetime_column_name: str, the name of the datetime column in the DataFrame
    - target_months: list of integers representing months to include
    Returns:
    - DataFrame with missing hours filled, NaN in numeric columns, string in string columns
    """

    # Convert the datetime column to datetime type if it's not already
    df[datetime_column_name] = pd.to_datetime(df[datetime_column_name])
    # Generate a complete hourly range based on the minimum and maximum datetimes in the DataFrame
    complete_range = pd.date_range(start=df[datetime_column_name].min(), end=df[datetime_column_name].max(), freq='H')
    # Create a DataFrame with the complete hourly range
    complete_df = pd.DataFrame({datetime_column_name: complete_range})
    # fill it with the constants too
    for col in constant_columns:
        complete_df[col] = df.reset_index()[col][0]
    # Filter complete DataFrame based on target months
    complete_df = complete_df[complete_df[datetime_column_name].dt.month.isin(target_months)]
    # Merge the complete DataFrame with the existing DataFrame
    merged_df = pd.merge(complete_df, df, on=[datetime_column_name]+constant_columns, how='left')
    return merged_df

baseDir = r'C:\Users\wkeenan\OneDrive - DOI\Documents\GitHub\Ozone_ML'
# Import data
O3J = pd.read_csv(fr"{baseDir}/Year2/Merged_Data/merge7.csv")
# do some preprocessing
# remove columns
remove = []
for col in O3J.columns:
    if 'Unnamed' in col or 'pressure' in col:
        remove.append(col)
O3J = O3J.drop(columns=remove)
# rename ozone
O3J.rename(columns={'sample_measurement':'o3'}, inplace=True)
# make columns for day and hour of day
O3J['datetime'] = pd.to_datetime(O3J['datetime'], utc=False)
O3J['datetime'] = O3J['datetime'].dt.tz_localize(None)
# O3J.set_index('datetime', inplace=True)
# O3J.index = O3J['datetime'].tz_convert('America/Denver')

# remove values that are negative. They will be interpolated later
O3J['o3'].where(O3J['o3'] >= 0, other=np.nan, inplace=True)
# also remove values that are over 200 ppb. They will be interpolated later
O3J['o3'].where(O3J['o3'] < 0.2, other=np.nan, inplace=True)
#fifthP = O3J['o3'].quantile(q=0.05)
# fill missing hours
dfs = dict(tuple(O3J.groupby('site_name')))
new_dfs = []
# decide your timesize
timesize = 96
# columns we care about interpolating
cols = ['o3', 'no2', 't2m', 'r2', 'sp', 'dswrf', 'MAXUVV', 'MAXDVV', 'orog', 'u10', 'v10', 'day_of_year', 'hour_of_day','pop_den'] #, 'no2', 'no2_bool'
for adf in dfs.values():

    df = fill_missing_hours(adf, 'datetime', target_months=[5,6,7,8,9], constant_columns=['county_code', 'site_number', 'county', 'site', 'site_name', 'pop_den', 'no2_bool'])
    df = add_time_columns(df, 'datetime')
    df = df[df['datetime'] > dt.datetime(year=2021, month=5, day=4)]

    site = df.reset_index()['site_name'][0]
    print(site)

    for col in cols:
        # interpolate everything. After this there should be no NANs that are timesize hours away from other NaNs
        df[col] = df[col].interpolate() # limit=timesize

    df.rename(columns={'datetime':'actual_datetime'}, inplace=True)
    df['datetime'] = pd.date_range(start=df['actual_datetime'].min(), periods=len(df), freq='H')

    df.to_csv(fr'{baseDir}\Year2\Merged_Data\nh\{site}.csv')
    constant_columns = ['orog', 'pop_den']

sites = ['Evergreen', 'Idaho Springs', 'Five Points', 'Welby', 'Highlands Ranch', 'Rocky Flats', 'Boulder', 'Chatfield Reservoir', 'Sunnyside', 'East Plains', 'South Table']

for site in sites: