import datetime as dt
import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tensorflow.keras.layers import Input, LSTM,Dense, Dropout
from tensorflow.keras import layers, optimizers, losses, metrics, Model
from sklearn.linear_model import LinearRegression
import math
from math import log10, floor, log
import scipy.stats
import seaborn as sn


# this function normalizes all the variables to be between zero and 1
def normalize(df, cols):
    df2 = df.copy()
    for col in cols:
        mini = df2[col].min()
        maxi = df2[col].max()
        rang = maxi - mini
        df2[col] = (df2[col] - mini) / rang
    return df2


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

    input_var_cnt = len(cols) ##the number of variables used to perform prediction
    ind_arr = np.zeros(shape=((len(df)),timesize,input_var_cnt)) # there are x variables and we are looking y timesteps back. Each input is a 2D array of data. We store all these in this 3D array
    dep_arr = np.zeros(shape=((len(df)))) # just a 1D array of discharge. It's the dependent varaible we train the model on
    ind_arr[:] = np.nan
    dep_arr[:] = np.nan
    dateArray = []
    for index, row in df.iterrows():
        if index < (len(df) - timesize):
            input = df[cols][index:index+timesize].to_numpy()
            ind_arr[index] = input # add a 2D array to our 3D array
            dep_arr[index] = df[dep_var][index+timesize]
            dateArray.append(str(df['date'][index+timesize]))

    # remove nans
    ind_arr = remove_nan_bands(ind_arr)
    dep_arr = dep_arr[~np.isnan(dep_arr)]
    return ind_arr, dep_arr

def prepare_lstm_data(df, dependent_variable, independent_variables, t):
    """
    Prepares data for training an LSTM model.
    Parameters:
    - df: pandas DataFrame
      Input dataframe with hourly timestamps as index.
    - dependent_variable: str
      Name of the dependent variable column.
    - independent_variables: list of str
      List of names of independent variable columns.
    - t: int
      Timestep, number of hours to consider in each sequence.
    Returns:
    - X: numpy array
      3D array of independent variables.

    - y: numpy array
      1D array of dependent variable.
    """
    # Ensure the dataframe is sorted by the timestamp
    df = df.sort_index()

    # Extract dependent variable
    y = df[dependent_variable].values

    # Extract independent variables
    X = []

    for i in range(len(df) - t + 1):
        # Check if there are any gaps in time (missing months)
        if (df.index[i + t - 1] - df.index[i]).days == t - 1:
            # Select timesteps for each sequence
            sequence = df[independent_variables].iloc[i:i + t].values
            X.append(sequence)

    # Convert the lists to numpy arrays
    X = np.array(X)
    y = y[t - 1:]  # Adjust y to align with X

    return X, y

X, y = prepare_lstm_data(dfs['Boulder'], 'sample_measurement', cols, 6)
dfs['Boulder'] = dfs['Boulder'].set_index('datetime')
def remove_nan_bands(input_array):
    # Check if all values in each stack are nan
    nan_mask = np.all(np.isnan(input_array), axis=(1, 2))
    # Remove stacks with all nan values
    result_array = input_array[~nan_mask, :, :]
    return result_array

# Import data
O3J = pd.read_csv(r"C:\Users\willy\Documents\GitHub\Ozone_ML\Year2\Merged_Data\merge3.csv")
cols = ['sample_measurement','t2m', 'r2', 'sp', 'dswrf', 'MAXUVV', 'MAXDVV', 'u10', 'v10']


corr_matrix = O3Jn.corr()
sn.heatmap(corr_matrix, annot=True)
plt.show()
# normalize the data
O3Jn, sdict, mdict = normalizeZ(O3J, cols)


# split into training, test, and validation sets a different way
O3J['datetime'] = pd.to_datetime(O3J['datetime']).dt.date
# train it on 2021 and 2022
training = O3J[O3J['datetime'] < dt.date(year=2023, month=1, day=1)]
test = [65, 27, 57, 7, 11] # testing sites
val = [50, 70, 33, 3002, 39, 73] # validation sites
validation = O3J[(O3J['datetime'] > dt.date(year=2023, month=1, day=1)) & (O3J['site'].isin(val))]
testing = O3J[(O3J['datetime'] > dt.date(year=2023, month=1, day=1)) & (O3J['site'].isin(test))]

sets = [training, validation, testing]

trainIAs = {}
trainDAs = {}
tIAs = {}
tDAs = {}
vIAs = {}
vDAs = {}
for s in sets:
    dfs = dict(tuple(s.groupby('site_name')))
    for df in dfs:
        df = df.reset_index()
        site = df['site_name'][0]
        print(site)
        df.sort_values(by='datetime', inplace=True)
        df = df.interpolate(limit=6)
        df['datetime'] = pd.to_datetime(df['datetime'])
        # add site column for now
        cols = ['sample_measurement', 't2m', 'r2', 'sp', 'dswrf', 'MAXUVV', 'MAXDVV', 'u10', 'v10', 'site']
        train_ia, train_da, test_ia, test_da = dataPrep(df, 26, cols, 'sample_measurement', date2split=pd.Timestamp(year=2023, day=5, month=1, tz='utc'))
        trainIAs[site] = train_ia
        trainDAs[site] = train_da
        tvIAs[site] = test_ia
        tvDAs[site] = test_da

trainIA = np.vstack(list(trainIAs.values()))
trainDA = np.hstack(list(trainDAs.values()))

'''configure the model'''
timesize = 6 ##e.g., past 6 hours
input_var_cnt = 9 ##the number of variables used to perform prediction e.g., NO2, Ozone ... from the previous x tine steps
input_lstm = Input(shape=(timesize, input_var_cnt)) ##what is the input for every sample, sample count is not included every sample should be a 2D matrix
##prepare a LSTM layer
unit_lstm = 32 ##hidden dimensions transfer data, larger more complex model
lstmlayer = LSTM(unit_lstm) (input_lstm) ##this outputs a matrix of 1*unit_lstm, the format is the layer (input), the output of the layer stores the time series info and the interaction of variables..
denselayer = Dense(1)(lstmlayer) ## reduce the hidden dimension to 1 ==== output data ,1 value for 1 input sample --- predicted ozone
model = Model(inputs = input_lstm, outputs = denselayer)
model.compile(loss='mse', optimizer='adam') ##how to measure the accuracy  compute mean squared error using all y_pred, y_true

# Loop through stations, run LSTM on data from each one, run regression on the results comparing actual vs expected value
stations = []
outputs = []
trains = []
dtownDen = [float(39.751184)]
results = {}

# I could loop through each ozone monitoring site, but I'll just loop through one site for an example
#for station in test['latitude'].unique():
for station in dtownDen:
    oneStation = O3J[O3J['latitude'] == station]
    stations.append(station)
    oneStation.dropna(inplace=True)
    oneStation.reset_index(inplace=True)

    inputArray = np.zeros(shape=((len(oneStation)),6,9))
    outputArray = np.zeros(shape=((len(oneStation))))
    for index, row in oneStation.iterrows():
        if index < (len(oneStation) - 6):
            input = oneStation[['sample_measurement','t2m', 'r2', 'sp', 'dswrf', 'MAXUVV', 'MAXDVV', 'u10', 'v10']][index:index+6].to_numpy()
            inputArray[index] = input
            outputArray[index] = oneStation['sample_measurement'][index+6]

    model.fit(inputArray, outputArray, epochs=100, batch_size=32, verbose=2)
    trainPredict = model.predict(inputArray)

    outputs.append(outputArray)
    trains.append(trainPredict)

def runRegression(xvars, y):
    X = xvars.reshape(-1,1)
    y = y
    model1 = LinearRegression()
    model1.fit(X, y)

    r_sq = model1.score(X, y)
    print(f"coefficient of determination: {r_sq}")
    #print('coefficients '+ str(model1.coef_))
    #print('intercept '+ str(model1.intercept_))
#
for i in range(len(outputs)):
    print('latitude of station{}'.format(stations[i]))
    runRegression(xvars=outputs[i], y=trains[i])

