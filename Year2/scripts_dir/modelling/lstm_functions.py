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

def trainLSTMgpt(ia, da, epochs=25, units=64, drop=0.2, batch=64, layers=0):
    model = Sequential()
    if layers > 0:
        for i in range(layers):
            model.add(LSTM(units=units, input_shape=(ia.shape[1], ia.shape[2]), return_sequences=True))
        model.add(LSTM(units=units, input_shape=(ia.shape[1], ia.shape[2])))
    else:
        model.add(LSTM(units=units, input_shape=(ia.shape[1], ia.shape[2])))
    model.add(Dropout(drop))  # Adding 20% dropout
    model.add(Dense(units=1))  # Output layer with 1 neuron for regression task
    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    # Train the model
    model.fit(ia, da, epochs=epochs, batch_size=batch)  # Adjust epochs and batch_size as needed
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

    rsq = model1.score(X, y)
    print(f"r2: {rsq}")

    rms = mean_squared_error(actual, pred, squared=False)
    print(f"rmse: {rms}")

    return rms, rsq

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

def multistep_forecast(model, hours_ahead, input_ia, input_da, input_dates, ozone_column, rdict, mdict):
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
    # Validate ozone_column
    if ozone_column < 0 or ozone_column >= input_ia.shape[2]:
        raise ValueError("Invalid value for ozone_column.")
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
    metrics_rms = {}
    metrics_rsq = {}
    for i in range(hours_ahead):
        print(f'predicting {i} hours ahead')
        # test for time continuity. Some arrays will not be time continuous because of missing data. Remove them.
        bad_times = time_continuity_test(input_sequence, 10, 11, rdict, mdict)
        # remove bad times
        input_sequence = input_sequence[bad_times]
        nextDA = nextDA[bad_times]
        next_dates = next_dates[bad_times]
        # Predict one step ahead
        predicted_step = model.predict(input_sequence)
        # Update the input sequence for the next prediction
        input_sequence = next_prediction(input_sequence, predicted_step, 0)
        # unNormalize them so you can evaluate
        predictionsUn = unNormalize(predicted_step, rdict['o3'], mdict['o3'])
        daUn = unNormalize(nextDA, rdict['o3'], mdict['o3'])
        # evaluate
        rms, rsq = evaluate(predictionsUn, daUn)
        metrics_rms[f'rms{i}'] = rms
        metrics_rsq[f'rsq{i}'] = rsq
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
    return predictions, actuals, dates, metrics_rms, metrics_rsq

def next_prediction(input_array, predictions, column):
    """
    :param input_array: the independent array from the previous timestep
    :param predictions: the predictions produces from the input array
    :param column: integer, the column of the input array the ozone lives in
    :return:
    """
    next = input_array.copy()
    for band in range(input_array.shape[0]-1):
        next_hour = input_array[band+1][-1]
        next_hour[column] = predictions[band]
        next_band = np.concatenate((next[band], [next_hour]))
        next_band = np.delete(next_band, 0, 0)
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

def time_continuity_test(ia, day_col, hour_col, rdict, mdict):
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
    if len(bad_dates) == 0:
        print('no bad dates found')
    bad_times = boolean_array_from_3d_with_false(ia, bad_dates)
    return bad_times

# bad_times = time_continuity_test(vIAs_f_24['Evergreen'], 10, 11)
# lastDay, lastHour = time_continuity_test(vIAs_f_24['Evergreen'], 10, 11)
def evalModel(model, valIA, valDA):
    predictedO3 = model.predict(valIA)
    predictedO3un = unNormalize(predictedO3, rdict['o3'], mdict['o3'])
    vDA_un = unNormalize(valDA, rdict['o3'], mdict['o3'])
    evaluate(predictedO3un, vDA_un)
    return predictedO3un, vDA_un

def split_data(O3Jn, sets, cols, one_hot, timesize):
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
