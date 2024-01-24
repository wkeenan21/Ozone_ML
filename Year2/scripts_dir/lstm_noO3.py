import datetime as dt
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
    ind_arr[:] = np.nan
    dep_arr[:] = np.nan
    df = df.reset_index(drop=True)
    dateArray = []
    for index, row in df.iterrows():
        if index < (len(df) - timesize):
            input = df[cols][index:index+timesize].to_numpy()
            ind_arr[index] = input # add a 2D array to our 3D array
            dep_arr[index] = df[dep_var][index+timesize]
            dateArray.append(str(df['datetime'][index+timesize]))

    ind_arr2, nansia = remove_bands_with_nans(ind_arr)
    nansda = np.isnan(dep_arr)
    nans = nansia + nansda

    dep_arr = dep_arr[~nans]
    ind_arr = ind_arr[~nans]

    if has_nan(dep_arr) or has_nan(ind_arr):
        raise ValueError('fuck')
    # dr = np.array(dateArray)
    # dr = dr[~nans]

    # # remove nans
    # ind_arr = remove_nan_bands(ind_arr)
    # dep_arr = dep_arr[~np.isnan(dep_arr)]
    return ind_arr, dep_arr

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

def evaluate(actual, pred):
    X = actual.reshape(-1,1)
    y = pred
    model1 = LinearRegression()
    model1.fit(X, y)

    r_sq = model1.score(X, y)
    print(f"r2: {r_sq}")

    rms = mean_squared_error(actual, pred, squared=False)
    print(f"rmse: {rms}")
    #print('coefficients '+ str(model1.coef_))
    #print('intercept '+ str(model1.intercept_))

def fill_missing_hours(df, datetime_column_name, target_months):
    """
    Fill missing hours in a DataFrame by adding rows for every hour in the range.

    Parameters:
    - df: pandas DataFrame
    - datetime_column_name: str, the name of the datetime column in the DataFrame
    - target_months: list of integers representing months to include
    Returns:
    - DataFrame with missing hours filled, NaN in other columns
    """
    # Convert the datetime column to datetime type if it's not already
    df[datetime_column_name] = pd.to_datetime(df[datetime_column_name])
    # Generate a complete hourly range based on the minimum and maximum datetimes in the DataFrame
    complete_range = pd.date_range(start=df[datetime_column_name].min(), end=df[datetime_column_name].max(), freq='H')
    # Create a DataFrame with the complete hourly range
    complete_df = pd.DataFrame({datetime_column_name: complete_range})
    # Filter complete DataFrame based on target months
    complete_df = complete_df[complete_df[datetime_column_name].dt.month.isin(target_months)]
    # Merge the complete DataFrame with the existing DataFrame
    merged_df = pd.merge(complete_df, df, on=datetime_column_name, how='left')
    return merged_df

def multistep_forecast(model, hours_ahead, input_ia, input_da, ozone_column):
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
    # Perform multistep forecasting
    predictions = []
    actuals = []
    for _ in range(hours_ahead):
        print(len(input_sequence), len(nextDA))
        # Predict one step ahead
        predicted_step = model.predict(input_sequence)
        # Append the predicted step to the results
        predictions.append(predicted_step)
        # Update the input sequence for the next prediction
        input_sequence = next_prediction(input_sequence, predicted_step, 0)
        # Update the ozone column with the predicted value
        #input_sequence[:, -1, ozone_column] = predicted_step
        # get the next DA
        evaluate(predicted_step, nextDA)
        actuals.append(nextDA)
        nextDA = next_DA(nextDA)
    return predictions, actuals

def next_prediction(input_array, predictions, column):
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
O3J = add_time_columns(O3J, 'datetime')
# remove values that are zero (never 0 ozone in the atmosphere)
fifthP = O3J['o3'].quantile(q=0.05)
O3J['o3'].where(O3J['o3'] > 0, other=fifthP, inplace=True)
# normalize the data
cols = ['o3','t2m', 'r2', 'sp', 'dswrf', 'MAXUVV', 'MAXDVV', 'u10', 'v10', 'day_of_year', 'hour_of_day']
O3Jn, rdict, mdict = normalize(O3J, cols)

# split into training, test, and validation sets a different way
O3Jn['date'] = pd.to_datetime(O3Jn['datetime']).dt.date
# train it on 2021 and 2022
training = O3Jn[O3Jn['date'] < dt.date(year=2023, month=1, day=1)]
test = [65, 27, 57, 7, 11] # testing sites
val = [50, 70, 33, 3002, 39, 73] # validation sites
validation = O3Jn[(O3Jn['date'] > dt.date(year=2023, month=1, day=1)) & (O3Jn['site'].isin(val))]
testing = O3Jn[(O3Jn['date'] > dt.date(year=2023, month=1, day=1)) & (O3Jn['site'].isin(test))]

sets = [training, validation, testing]
trainIAs = {}
trainDAs = {}
tIAs = {}
tDAs = {}
vIAs = {}
vDAs = {}

count = 0
for s in sets:
    dfs = dict(tuple(s.groupby('site_name')))
    count+=1
    for df in dfs.values():
        df = df.reset_index()
        site = df['site_name'][0]
        print(site)
        # sort by datetime so the interpolation makes sense
        df.sort_values(by='datetime', inplace=True)
        df = fill_missing_hours(df, 'datetime', target_months=[5,6,7,8,9])
        # interpolate everything. After this there should be no NANs that are 72 hours away from other NaNs
        cols = ['o3', 't2m', 'r2', 'sp', 'dswrf', 'MAXUVV', 'u10', 'v10']
        for col in cols:
            df[col] = df[col].interpolate(limit=24)
        # df['datetime'] = pd.to_datetime(df['datetime'])
        # add site column for now

        ia, da, = dataPrep(df, 24, cols, 'o3')

        if np.isnan(ia).any() or np.isnan(da).any():
            raise ValueError('explode')
        if count == 1: # just verifies this is the training data
            trainIAs[site] = ia
            trainDAs[site] = da
        elif count == 2:
            vIAs[site] = ia
            vDAs[site] = da
        else:
            tIAs[site] = ia
            tDAs[site] = da

trainIA = np.vstack(list(trainIAs.values()))
trainDA = np.hstack(list(trainDAs.values()))

vIA = np.vstack(list(vIAs.values()))
vDA = np.hstack(list(vDAs.values()))

tIA = np.vstack(list(tIAs.values()))
tDA = np.hstack(list(tDAs.values()))

# drop ozone from training data and see how it does
trainIA_noO3 = np.delete(trainIA, 0, axis=2)
vIA_noO3 = np.delete(vIA, 0, axis=2)

#model = runLSTM(trainIA, trainDA, 72, cols=cols, activation='sigmoid', epochs=25, units=32, run_model=True)
model = trainLSTMgpt(trainIA_noO3, trainDA)

# see how it did on validation for all the places
predictedO3 = model.predict(vIA_noO3)
predictedO3un = unNormalize(predictedO3, rdict['o3'], mdict['o3'])
vDA_un = unNormalize(vDA, rdict['o3'], mdict['o3'])
evaluate(predictedO3un, vDA_un)

# see how it did just for south table
st_vIA = vIAs['South Table'].copy()
st_vDA = vDAs['South Table'].copy()
st_predictedO3 = model.predict(st_vIA)
st_predictedO3un = unNormalize(st_predictedO3, rdict['o3'], mdict['o3'])
st_vDA_un = unNormalize(st_vDA, rdict['o3'], mdict['o3'])
evaluate(st_predictedO3un, st_vDA_un)

# multistep test
preds, actuals = multistep_forecast(model, 10, st_vIA, st_vDA, 0)





plt.scatter(predictedO3, vDA_un, color='blue', marker='o', label='Scatter Plot')

# Set the maximum values for x and y axes
plt.xlim(0, 0.1)
plt.ylim(0, 0.1)
# Add labels and title
plt.xlabel('X-axis Label')
plt.ylabel('Y-axis Label')
plt.title('Scatter Plot Example')

# Add a legend
plt.legend()

# Show the plot
plt.show()


