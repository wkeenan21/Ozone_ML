import datetime as dt
from datetime import timedelta
import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM,Dense, Dropout
from tensorflow.keras import layers, optimizers, losses, metrics, Model
from sklearn.linear_model import LinearRegression
import math
from math import log10, floor, log
import scipy.stats
import sklearn.preprocessing as sk
#import seaborn as sn
from sklearn.metrics import mean_squared_error



# this function normalizes all the variables to be between zero and 1
def normalize(df, cols):
    df2 = df.copy()
    rangDict = {}
    minDict = {}
    for col in cols:
        mini = df2[col].min()
        maxi = df2[col].max()
        rang = maxi - mini
        df2[col] = (df2[col] - mini) / rang
        rangDict[col] = rang
        minDict[col] = mini
    return df2, rangDict, minDict

def unNormalize(ar, rang, mini):
    ar2 = (ar * rang) + mini
    return ar2

# this function normalizes all the variables to be between zero and 1 using Z scores
def normalizeZ(df, cols):
    df2 = df.copy()
    sDict = {}
    mDict = {}
    for col in cols:
        mean = df2[col].mean()
        std = df2[col].std()
        df2[col] = (df2[col] - mean) / std
        sDict[col] = std
        mDict[col] = mean

    return df2, sDict, mDict

def unNormalize_df(df, cols, mean, std):
    # unnormalizes a df on the columns you give it
    df2 = df.copy()
    for i in range(len(cols)):
        df2[cols[i]] = (df2[cols[i]] * std) + mean
    return df2

def unNormalize_ar(ar, mean, std):
    # unnormalize an array
    ar2 = (ar * std) + mean
    return ar2

# where the magic happens
def runLSTM(ind_arr, dep_arr, timesize, cols, activation, epochs, units=10, run_model=True):
    '''configure the model'''
    input_var_cnt = len(cols) ##the number of variables used to perform prediction
    input_lstm = Input(shape=(timesize, input_var_cnt)) ##what is the input for every sample, sample count is not included every sample should be a 2D matrix
    ##prepare a LSTM layer
    unit_lstm = units ##hidden dimensions transfer data, larger more complex model
    lstmlayer = LSTM(unit_lstm, activation=activation) (input_lstm) ##this outputs a matrix of 1*unit_lstm, the format is the layer (input), the output of the layer stores the time series info and the interaction of variables..
    denselayer = Dense(1)(lstmlayer) ## reduce the hidden dimension to 1 ==== output data ,1 value for 1 input sample
    model = Model(inputs = input_lstm, outputs = denselayer)
    model.add(Dropout(0.2)) # 20% of the neurons get ignored at random times to prevent overfitting
    model.compile(loss='mse', optimizer='adam') ##how to measure the accuracy  compute mean squared error using all y_pred, y_true

    # now fit the model
    if run_model:
        model.fit(ind_arr, dep_arr, epochs=epochs, batch_size=32, verbose=2)
        return model

def trainLSTMgpt(ia, da, epochs=25):
    model = Sequential()
    model.add(LSTM(units=32, input_shape=(ia.shape[1], ia.shape[2])))
    model.add(Dropout(0.2))  # Adding 20% dropout
    model.add(Dense(units=1))  # Output layer with 1 neuron for regression task

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(ia, da, epochs=epochs, batch_size=32)  # Adjust epochs and batch_size as needed
    return model

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

# this preps the data. It takes a dataframe and splits it by date into training and testing sets. It also splits it into dependent and independent variables for each.
def dataPrep_wsplit(df, timesize, cols, dep_var, date2split):
    input_var_cnt = len(cols) ##the number of variables used to perform prediction
    ind_arr = np.zeros(shape=((len(df)),timesize,input_var_cnt)) # there are x variables and we are looking y timesteps back. Each input is a 2D array of data. We store all these in this 3D array
    dep_arr = np.zeros(shape=((len(df)))) # just a 1D array of discharge. It's the dependent varaible we train the model on
    ind_arr[:] = np.nan
    dep_arr[:] = np.nan
    dateArray = []
    split = None
    for index, row in df.iterrows():
        if index < (len(df) - timesize):
            input = df[cols][index:index+timesize].to_numpy()
            ind_arr[index] = input # add a 2D array to our 3D array
            dep_arr[index] = df[dep_var][index+timesize]
            dateArray.append(str(df['datetime'][index+timesize]))
            if not split: # prevent it from finding more splits
                if date2split <= df['datetime'][index]:
                    split = index - timesize # split training and test on this row
                    print(f"split on {df['datetime'][split]}")

    train_ind_arr = ind_arr[0:split]
    test_ind_arr = ind_arr[split:]
    train_dep_arr = dep_arr[0:split]
    test_dep_arr = dep_arr[split:]
    # remove nans
    train_ind_arr = remove_nan_bands(train_ind_arr)
    test_ind_arr = remove_nan_bands(test_ind_arr)
    train_dep_arr = train_dep_arr[~np.isnan(train_dep_arr)]
    test_dep_arr = test_dep_arr[~np.isnan(test_dep_arr)]
    return train_ind_arr, train_dep_arr, test_ind_arr, test_dep_arr

def dataPrep(df, timesize, cols, dep_var):
    """
    :param df: dataframe you want to prep
    :param timesize: how many rows back you want to look
    :param cols: just used for number of columns
    :param dep_var: which of the columns is also the dependent variable
    :return: 2 arrays, plus a date array of the selected datetimes
    """
    input_var_cnt = len(cols) ##the number of variables used to perform prediction
    ind_arr = np.zeros(shape=((len(df)),timesize,input_var_cnt)) # there are x variables and we are looking y timesteps back. Each input is a 2D array of data. We store all these in this 3D array
    dep_arr = np.zeros(shape=((len(df)))) # just a 1D array of discharge. It's the dependent varaible we train the model on
    dateArray = np.zeros(shape=((len(df)))) # an array of dates to keep track of things
    ind_arr[:] = np.nan
    dep_arr[:] = np.nan
    dateArray = dateArray.astype(str)
    df = df.reset_index(drop=True)
    for index, row in df.iterrows():
        if index < (len(df) - timesize):
            input = df[cols][index:index+timesize].to_numpy()
            ind_arr[index] = input # add a 2D array to our 3D array
            dep_arr[index] = df[dep_var][index+timesize]
            dateArray[index] = df['datetime'][index+timesize]

    ind_arr2, nansia = remove_bands_with_nans(ind_arr)
    nansda = np.isnan(dep_arr)
    nans = nansia + nansda
    # remove nans
    dep_arr = dep_arr[~nans]
    ind_arr = ind_arr[~nans]
    dateArray = dateArray[~nans]

    if has_nan(dep_arr) or has_nan(ind_arr):
        raise ValueError('fuck')
    # dr = np.array(dateArray)
    # dr = dr[~nans]

    # # remove nans
    # ind_arr = remove_nan_bands(ind_arr)
    # dep_arr = dep_arr[~np.isnan(dep_arr)]
    return ind_arr, dep_arr, dateArray

def remove_indices_from_list(original_list, indices_to_remove):
    # Create a new list excluding the specified indices
    modified_list = [item for index, item in enumerate(original_list) if index not in indices_to_remove]
    return modified_list

def find_nan_indices(array_3d):
    # Get the indices where NaN values are present along the 0-axis
    nan_indices_0_axis = np.where(np.isnan(array_3d[0, :, :]))[0]
    nan_indices_0_axis = list(set(nan_indices_0_axis))

    return nan_indices_0_axis


def remove_bands_with_nans(array_3d):
    # Find the bands (depth slices) with any NaN values
    bands_with_nans = np.isnan(array_3d).any(axis=(1, 2))

    # Use boolean indexing to keep only the bands without NaN values
    filtered_array_3d = array_3d[~bands_with_nans]

    return filtered_array_3d, bands_with_nans


def remove_indices_from_depth(array_3d, indices_to_remove):
    # Check if indices are within valid range
    if max(indices_to_remove) >= array_3d.shape[0]:
        raise ValueError("Invalid indices to remove. Indices exceed depth dimension.")

    # Create a new array excluding the specified indices along the depth dimension
    modified_array_3d = np.delete(array_3d, indices_to_remove, axis=0)

    return modified_array_3d

def has_nan(array):
    # Check if any element in the array is NaN
    return np.isnan(array).any()

def remove_nan_bands(input_array):
    # Check if all values in each stack are nan
    nan_mask = np.all(np.isnan(input_array), axis=(1, 2))
    # Remove stacks with all nan values
    result_array = input_array[~nan_mask, :, :]
    return result_array

def nse(predictions, targets):
    return (1-(np.sum((targets-predictions)**2)/np.sum((targets-np.mean(targets))**2)))

def evaluate(actual, pred):
    X = actual.reshape(-1,1)
    y = pred
    model1 = LinearRegression()
    model1.fit(X, y)

    r_sq = model1.score(X, y)
    print(f"r2: {r_sq}")

    rms = mean_squared_error(actual, pred, squared=False)
    print(f"rmse: {rms}")

    #nseVal = nse(pred, actual)
    #print('coefficients '+ str(model1.coef_))
    #print('intercept '+ str(model1.intercept_))

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

def multistep_forecast(model, hours_ahead, input_ia, input_da, input_dates, ozone_column=None):
    """
    Perform multistep forecasting using an LSTM model.
    Parameters:
    - model: Keras model object
    - hours_ahead: Number of hours to predict ahead
    - input_data: 3D matrix of independent variables
    - ozone_column: Index of the column containing ozone data
    Returns:
    - predictions: 1D array of predicted ozone values
    """
    # Validate input_data shape
    if len(input_ia.shape) != 3:
        raise ValueError("Input data must be a 3D matrix.")
    # Number of time steps in the input data
    time_steps = input_ia.shape[1]
    # Validate hours_ahead
    if hours_ahead <= 0 or hours_ahead > time_steps:
        raise ValueError("Invalid value for hours_ahead.")
    # Initial input sequence for prediction
    input_sequence = input_ia.copy()
    # initial DA
    nextDA = input_da.copy()
    # initial dates
    next_dates = input_dates.copy()
    # Perform multistep forecasting
    predictions = []
    actuals = []
    dates = []
    for _ in range(hours_ahead):
        # test for time continuity. Some arrays will not be time continuous because of missing data. Remove them.
        bad_times = time_continuity_test(input_sequence, -2, -1) # this assumes the day col and the hour col are the last and 2nd to last columns
        # remove bad times
        input_sequence = input_sequence[bad_times]
        nextDA = nextDA[bad_times]
        next_dates = next_dates[bad_times]
        # Predict one step ahead
        print(f'predicting {_+1} hours ahead')
        print(input_sequence.shape)
        predicted_step = model.predict(input_sequence)
        # Update the input sequence for the next prediction
        if ozone_column:
            input_sequence = next_prediction(input_sequence, predicted_step, ozone_column)
        else:
            input_sequence = next_prediction(input_sequence)
        # unNormalize them so you can evaluate
        predictionsUn = unNormalize(predicted_step, rdict['o3'], mdict['o3'])
        daUn = unNormalize(nextDA, rdict['o3'], mdict['o3'])
        # evaluate
        evaluate(predictionsUn, daUn)
        # append actuals to list
        actuals.append(nextDA)
        # Append the predicted step to the results
        predictions.append(predicted_step)
        # append the dates
        dates.append(next_dates)
        # get next DA
        nextDA = next_DA(nextDA)
        # get next dates
        next_dates = next_DA(next_dates)
    return predictions, actuals, dates

def next_prediction(input_array, predictions=None, column=None):
    """
    if no predictions or column, you are not using ozone as a predictor in the model
    :param input_array: the independent array from the previous timestep
    :param predictions: the predictions produces from the input array
    :param column: integer, the column of the input array the ozone lives in
    :return:
    """

    next = input_array.copy()
    for band in range(input_array.shape[0]-1):
        # grab the next hour of data in the sequence
        next_hour = input_array[band+1][-1]
        if column and predictions:
            # replace the actual ozone with the predicted ozone, use that to predict the next hour
            next_hour[column] = predictions[band]
        # add the next hour to the band, now next_band as timesize+1 rows
        next_band = np.concatenate((next[band], [next_hour]))
        # delete the first hour in the sequence so it's still using timesize hours of data
        next_band = np.delete(next_band, 0, 0)
        # reassign the band to the output
        next[band] = next_band
    next = np.delete(next, -1, 0)
    return next

def next_DA(inputDA):
    nextDA = inputDA[1:]
    return nextDA

def unNormalize(ar, rang, mini):
    ar2 = (ar * rang) + mini
    return ar2

def boolean_array_from_3d_with_false(arr, false_indexes):
    """
    Create a boolean array with True values along the first dimension of a 3D numpy array,
    except for specified indexes that should be set to False.
    Parameters:
    - arr: 3D numpy array
    - false_indexes: List of integers representing indexes to set to False
    Returns:
    - Boolean array with True values along the first dimension, False at specified indexes
    """
    # if not isinstance(arr, np.ndarray) or arr.ndim != 3:
    #     raise ValueError("Input must be a 3D numpy array.")
    # if not all(isinstance(idx, int) for idx in false_indexes):
    #     raise ValueError("false_indexes must be a list of integers.")
    # Create a boolean array with True values along the first dimension
    bool_array = np.ones(arr.shape[0], dtype=bool)
    # Set specified indexes to False
    bool_array[false_indexes] = False
    return bool_array


def combine_hour_day_to_datetime(hour_array, day_of_year_array, year):
    """
    Combine two 1D numpy arrays representing hour of the day and day of the year into a pandas Series of datetimes.
    Parameters:
    - hour_array: 1D numpy array representing hour of the day (values between 0 and 23)
    - day_of_year_array: 1D numpy array representing day of the year (values between 1 and 365)
    - year: Year to be used for the resulting datetimes
    Returns:
    - Pandas Series of datetimes
    """
    # if not isinstance(hour_array, np.ndarray) or not isinstance(day_of_year_array, np.ndarray):
    #     raise ValueError("Input must be 1D numpy arrays.")
    # if hour_array.shape != day_of_year_array.shape:
    #     raise ValueError("Arrays must have the same shape.")
    # if not np.all((hour_array >= 0) & (hour_array <= 23)):
    #     raise ValueError("Hour values must be between 0 and 23.")
    # if not np.all((day_of_year_array >= 1) & (day_of_year_array <= 365)):
    #     raise ValueError("Day of year values must be between 1 and 365.")
    # Create a DataFrame with columns for year, day of year, and hour
    df = pd.DataFrame({'year': year, 'day_of_year': day_of_year_array, 'hour': hour_array})

    # Convert to datetime
    df = create_month_day_columns(df, 'day_of_year')
    hours = pd.to_timedelta(hour_array, unit='H')
    df['datetime'] = pd.to_datetime(df[['month', 'day', 'year']])
    df['datetime'] = df['datetime'] + hours
    # Extract the resulting datetime Series
    datetime_series = pd.Series(df['datetime'])
    return datetime_series

def create_month_day_columns(df, day_of_year_column):
    # Extract month and day from the day of the year column
    df['month'] = pd.to_datetime(df[day_of_year_column], format='%j').dt.month
    df['day'] = pd.to_datetime(df[day_of_year_column], format='%j').dt.day
    return df

def is_sequential_hourly(series):
    """
    Check if a pandas Series of datetimes is sequential and does not skip hours.
    Parameters:
    - series: Pandas Series of datetimes
    Returns:
    - True if the datetimes are sequential without skipping hours, False otherwise
    """
    if not isinstance(series, pd.Series) or not pd.api.types.is_datetime64_any_dtype(series):
        raise ValueError("Input must be a pandas Series of datetimes.")
    # Check if the datetimes are hourly
    hourly_check = pd.to_timedelta(series.diff())[1:] == pd.Timedelta(hours=1)
    # Check if the datetimes are sequential without skipping hours
    is_sequential = hourly_check.all()
    return is_sequential

def time_continuity_test(ia, day_col, hour_col):
    bad_dates = []
    print('testing time continuity')
    for band in range(ia.shape[0]):
        days = unNormalize(ia[band,:, day_col], rdict['day_of_year'], mdict['day_of_year'])
        hours = unNormalize(ia[band,:,hour_col], rdict['hour_of_day'], mdict['hour_of_day'])
        days = days.round()
        hours = hours.round()
        series = combine_hour_day_to_datetime(hours, days, 2020)
        if not is_sequential_hourly(series):
            bad_dates.append(band)
    bad_times = boolean_array_from_3d_with_false(ia, bad_dates)
    return bad_times

# bad_times = time_continuity_test(vIAs_f_24['Evergreen'], 10, 11)
# lastDay, lastHour = time_continuity_test(vIAs_f_24['Evergreen'], 10, 11)


def split_data(sets, cols, one_hot, timesize):
    trainIAs = {}
    trainDAs = {}
    trainDates = {}
    tIAs = {}
    tDAs = {}
    tDates = {}
    vIAs = {}
    vDAs = {}
    vDates = {}
    count = 0
    df_list = []
    # add one hot encoding by site
    if one_hot:
        cols = cols + list(O3Jn['site_name'].unique())
    for s in sets:
        print(f'set {count}')
        dfs = dict(tuple(s.groupby('site_name')))
        df_list.append(dfs)
        count+=1
        for df in dfs.values():
            df = df.reset_index()
            site = df['site_name'][0]
            print(site)
            # sort it by time
            df = df.sort_values('datetime')
            # make the df only the columns we care about
            # cols = ['datetime','o3', 't2m', 'r2', 'sp', 'dswrf', 'MAXUVV', 'MAXDVV', 'orog', 'u10', 'v10', 'day_of_year','hour_of_day']
            # df = df[cols]

            # for col in cols:
            #     # interpolate everything. After this there should be no NANs that are timesize hours away from other NaNs
            #     df[col] = df[col].interpolate(limit=timesize)

            # split the independent and dependent arrays
            ia, da, dates = dataPrep(df, timesize, cols, 'o3')

            # if there's a single nan in either of these blow it up because it won't work
            if np.isnan(ia).any() or np.isnan(da).any():
                raise ValueError('explode')
            if count == 1: # just verifies this is the training data
                trainIAs[site] = ia
                trainDAs[site] = da
                trainDates[site] = dates
            elif count == 2:
                vIAs[site] = ia
                vDAs[site] = da
                vDates[site] = dates
            else:
                tIAs[site] = ia
                tDAs[site] = da
                tDates[site]= dates

    return trainIAs, trainDAs, trainDates, vIAs, vDAs, vDates, tIAs, tDAs, tDates, df_list


# Import data
O3J = pd.read_csv(r"D:\Will_Git\Ozone_ML\Year2\Merged_Data\merge3.csv")
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
# O3J.set_index('datetime', inplace=True)
# O3J.index = O3J['datetime'].tz_convert('America/Denver')

# remove values that are zero (never 0 ozone in the atmosphere)
fifthP = O3J['o3'].quantile(q=0.05)
O3J['o3'].where(O3J['o3'] > 0, other=fifthP, inplace=True)
# fill missing hours
dfs = dict(tuple(O3J.groupby('site_name')))
new_dfs = []
# decide your timesize
timesize = 24
# columns we care about interpolating
cols = ['o3','t2m', 'r2', 'sp', 'dswrf', 'MAXUVV', 'MAXDVV', 'orog', 'u10', 'v10', 'day_of_year', 'hour_of_day']
for adf in dfs.values():
    df = fill_missing_hours(adf, 'datetime', target_months=[5,6,7,8,9], constant_columns=['county_code', 'site_number', 'county', 'site', 'site_name'])
    df = add_time_columns(df, 'datetime')
    for col in cols:
        # interpolate everything. After this there should be no NANs that are timesize hours away from other NaNs
        df[col] = df[col].interpolate(limit=timesize)
    # append to list
    new_dfs.append(df)

O3J = pd.concat(new_dfs, ignore_index=True)

# normalize the data
O3Jn, rdict, mdict = normalize(O3J, cols)

#get one hot encoding
dummies = pd.get_dummies(O3Jn['site_name'])
O3Jn = pd.merge(O3Jn, dummies, left_index=True, right_index=True)

# split into training, test, and validation sets a different way
O3Jn['date'] = pd.to_datetime(O3Jn['datetime']).dt.date
# train it on 2021 and 2022
training = O3Jn[O3Jn['date'] < dt.date(year=2023, month=1, day=1)]
test = [65, 27, 57, 7, 11] # testing sites
val = [50, 70, 33, 3002, 39, 73, 65, 27, 57, 7, 11] # validation sites (using all as validation for now)
validation = O3Jn[(O3Jn['date'] > dt.date(year=2023, month=1, day=1)) & (O3Jn['site'].isin(val))]
testing = O3Jn[(O3Jn['date'] > dt.date(year=2023, month=1, day=1)) & (O3Jn['site'].isin(test))]

# create a list of sets
sets = [training, validation, testing]

# split the data by training, val, and testing
trainIAs_f_24, trainDAs_f_24, trainDates_f_24, vIAs_f_24, vDAs_f_24, vDates_f_24, tIAs_f_24, tDAs_f_24, tDates_f_24, dfs = split_data(sets, cols, False, 24)

# stack them into big arrays for training the universal model
trainIA_f_24 = np.vstack(list(trainIAs_f_24.values()))
trainDA_f_24 = np.hstack(list(trainDAs_f_24.values()))
vIA_f_24 = np.vstack(list(vIAs_f_24.values()))
vDA_f_24 = np.hstack(list(vDAs_f_24.values()))
tIA_f_24 = np.vstack(list(tIAs_f_24.values()))
tDA_f_24 = np.hstack(list(tDAs_f_24.values()))

trainIAs_t_24, trainDAs_t_24, trainDates_t_24, vIAs_t_24, vDAs_t_24, vDates_t_24, tIAs_t_24, tDAs_t_24, tDates_t_24, dfs2 = split_data(sets, cols, True, 24)
trainIA_t_24 = np.vstack(list(trainIAs_t_24.values()))
trainDA_t_24 = np.hstack(list(trainDAs_t_24.values()))
vIA_t_24 = np.vstack(list(vIAs_t_24.values()))
vDA_t_24 = np.hstack(list(vDAs_t_24.values()))
tIA_t_24 = np.vstack(list(tIAs_t_24.values()))
tDA_t_24 = np.hstack(list(tDAs_t_24.values()))
#ia, da, nans = split_data(sets, cols, False, 24)

#model = runLSTM(trainIA, trainDA, 72, cols=cols, activation='sigmoid', epochs=25, units=32, run_model=True)
#model_one_hot = trainLSTMgpt(trainIA_t_24, trainDA_t_24)

# drop ozone from training data and see how it does
trainIA_noO3 = np.delete(trainIA_f_24, 0, axis=2)
vIA_noO3 = np.delete(vIA_f_24, 0, axis=2)
model_no_one_hot = trainLSTMgpt(trainIA_noO3, trainDA_f_24)

# see how it did on validation for all the places
def evalModel(model, valIA, valDA):
    predictedO3 = model.predict(valIA)
    predictedO3un = unNormalize(predictedO3, rdict['o3'], mdict['o3'])
    vDA_un = unNormalize(valDA, rdict['o3'], mdict['o3'])
    evaluate(predictedO3un, vDA_un)
    #return predictedO3un, vDA_un

evalModel(model_no_one_hot, vIA_noO3, vDA_f_24)


# see how it did by site
merged_dfs = []
for site in vIAs_f_24.keys():
    print(site)

    lat = dfs[1][site]['latitude'].max()
    lon = dfs[1][site]['longitude'].max()
    st_vIA = vIAs_f_24[site].copy()
    st_vDA = vDAs_f_24[site].copy()
    # drop ozone from each
    st_vIA = np.delete(st_vIA, 0, axis=2)
    st_vDates = vDates_f_24[site].copy()
    # multistep test
    preds, actuals, dates = multistep_forecast(model_no_one_hot, 5, st_vIA, st_vDA, st_vDates, ozone_column=None)

    results = {}

    for i in range(len(preds)):
        results[i] = pd.DataFrame()
        results[i][f'date'] = dates[i]
        results[i][f'preds_{i}_{site}'] = preds[i].flatten()
    # add the actual o3 once
    results[0]['actual'] = actuals[0]

    from functools import reduce
    # Use functools.reduce and pd.merge to merge DataFrames on the 'date' column
    merged_df = reduce(lambda left, right: pd.merge(left, right, on='date'), results.values())
    merged_df['site_name'] = site
    merged_df['lat'] = lat
    merged_df['lon'] = lon
    merged_dfs.append(merged_df)
    #merged_df.to_csv(r"D:\Will_Git\Ozone_ML\Year2\results\{}_6hour_24time_n.csv".format(site))

merged_df = reduce(lambda left, right: pd.merge(left, right, on='date'), merged_dfs)