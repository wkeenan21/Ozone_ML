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
import datetime as dt
from datetime import timedelta
import hydroeval as he
import random

"""
First I define a bunch of functions I need
"""
# simple round to 1 digit function
def round_to_1(x):
    return round(x, -int(floor(log10(abs(x)))))
# plotting function for scatter plots. works without dates as xaxis
def plotScat(df, xaxis, yaxis, yaxisLabel=None, filt = None, yaxis2 = None, xaxisLabel = None, yaxis2Label = None, title = None, font='Times New Roman', fontsize=15, linreg=True, logreg=False, expreg=False, annotate_shift=0, saveDir=False):
    fig, ax = plt.subplots()
    tsfont = {'fontname': font, 'size': fontsize} # gotta add this as a kwarg to change the font on things, except legend
    fontParams = {'family' : font,'size': fontsize}
    matplotlib.rc('font', **fontParams) # this changes the font of the legend for some reason
    # filter is if we want to create multiple lines based on unique values of a column
    if filt != None:
        for year in df[filt].unique():
            filter = df[filt] == year
            ax.scatter(df[filter][xaxis], df[filter][yaxis], label=year)
        ax.set_xlabel(xaxis, **tsfont)
        ax.legend(frameon=True)
    else:
        ax.scatter(df[xaxis], df[yaxis], label='yaxis')

    if yaxis2 != None:
        ax2 = ax.twinx()
        ax2.scatter(df[xaxis], df[yaxis2], color='red')
        ax2.set_ylabel(yaxis2Label, **tsfont)

    if not yaxisLabel:
        yaxisLabel=yaxis
    ax.set_xlabel(xaxisLabel, **tsfont)
    ax.set_ylabel(yaxisLabel, **tsfont)

    # add regression line
    if linreg:
        m, b = np.polyfit(df[xaxis], df[yaxis], 1)
        plt.plot(df[xaxis], m * df[xaxis] + b, color='black')
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(df[xaxis], df[yaxis])
        if p_value < 0.001:
            p_value = '<0.01'
        else:
            p_value = str(round_to_1(p_value))
        ax.annotate('$R^2$ = ' + str(r_value**2)[0:4], (df[xaxis].median() +annotate_shift,df[yaxis].max()), **tsfont)

    # make the title
    plt.title(title, **tsfont)
    fig = plt.gcf()
    fig.set_size_inches(12, 7)
    if saveDir:
        plt.savefig(os.path.join(saveDir, yaxis + '_vs_' + xaxis + '.png'))
    plt.show()


# this function normalizes all the variables to be between zero and 1
def normalize(df, cols):
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

# simple linear regression
def runRegression(xvars, y):
    X = xvars.reshape(-1,1)
    model1 = LinearRegression()
    model1.fit(X, y)

    r_sq = model1.score(X, y)
    print(f"coefficient of determination: {r_sq}")
    return r_sq
    #print('coefficients '+ str(model1.coef_))
    #print('intercept '+ str(model1.intercept_))

def remove_nan_bands(input_array):
    # Check if all values in each stack are nan
    nan_mask = np.all(np.isnan(input_array), axis=(1, 2))
    # Remove stacks with all nan values
    result_array = input_array[~nan_mask, :, :]
    return result_array

# this preps the data. It takes a dataframe and splits it by date into training and testing sets. It also splits it into dependent and independent variables for each.
def dataPrep(df, timesize, cols, dep_var, date2split):
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
            dateArray.append(str(df['date'][index+timesize]))
            if date2split == df['date'][index]:
                split = index - timesize # split training and test on this row
                print(f"split on {df['date'][split]}")
    if not split:
        split = 75 # if that date2split doesn't exist for some reason just split on the 90th row
        print('split on 90')

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

# I use this to split the testing data into validation and test sets by gage
def extract_bands_by_watershed(input_Iarray, input_Darray, selected_watersheds):
    # input_Iarray must have gage IDs as first column for this to work
    # Extract the first column (watershed identifier) from each band
    watershed_identifiers = input_Iarray[:, 0, 0]

    # Find indices of bands with selected watershed identifiers
    selected_indices = np.isin(watershed_identifiers, selected_watersheds)

    # Extract bands with selected watershed identifiers
    selected_bands = input_Iarray[selected_indices]
    selected_Darray = input_Darray[selected_indices]

    # Extract watershed identifiers that are left
    remaining_indices = np.isin(watershed_identifiers, np.unique(watershed_identifiers[~selected_indices]))
    remaining_bands = input_Iarray[remaining_indices]
    remaining_Darray = input_Darray[remaining_indices]

    return selected_bands, selected_Darray, remaining_bands, remaining_Darray

# I use this to pick testing gages and validation gages.
def split_list_randomly(input_list):
    # Make a copy of the input list to avoid modifying the original list
    shuffled_list = input_list.copy()

    # Shuffle the list randomly
    random.shuffle(shuffled_list)

    # Calculate the midpoint to split the list into two halves
    midpoint = len(shuffled_list) // 2

    # Split the list into two halves
    list1 = shuffled_list[:midpoint]
    list2 = shuffled_list[midpoint:]

    return list1, list2

"""
Now prep the data. We need 4 arrays: training independent variables, training dependent variables, testing independent variables, testing dependent variables
"""
all_gage_list = ['06468170', '05336700', '08068090', '06918060', '06908000', '06481500', '01491000', '01578475',
                 '02131500', '11473900', '05078500', '09517000', '01580520', '07047950', '05388250', '06928000',
                 '05244000', '13305000', '06479525', '08110000', '05132000', '07290000', '08164000', '05131500',
                 '06076690', '02202500', '02135000', '05418500', '05300000', '13302005', '06815000', '02049500',
                 '02175000', '05422000', '05123400', '01594440', '09487000', '05062500', '09512800', '09485700',
                 '02198000', '12324680', '05412500', '07346070', '05066500', '11519500', '07169500', '06018500',
                 '06916600', '11348500', '01643000', '05304500', '11501000', '05313500', '06052500', '11517500',
                 '08117500', '08033500', '09439000', '05447500', '05090000', '02136000', '06471200', '05434500',
                 '07363500', '06821190', '11376000', '05057200', '07364200', '07288500', '09537500', '05056000']

dataFold = r"T:\ProjectWorkspace\EPA\SSWR_EPA_project\Gage_time_series\Will_Keenan\Merged_Data\v6_storage_calculations\w_geology"
cols = ['PR_sum','PRCP_sum','VPD_mean','TMMN_mean','TMMX_mean','ET_mean','SRAD_mean','SWE_mean','Q_mean']

# first get all the dfs into one big df, do some data cleaning
dfs = []
exclude = ['12324680', '11501000', '07290000', '05422000', '06468170', '08110000'] # these got problems
gages = list(set(all_gage_list) - set(exclude))
gages = [int(i) for i in gages] # it's just easier if they are all ints since dfs do that
for file in os.listdir(dataFold):
    gage = file[0:8]
    if gage not in exclude: # exclude the problem children
        path = os.path.join(dataFold, file)
        df = pd.read_csv(path)
        df.dropna(inplace=True)
        df['date'] = pd.to_datetime(df['date'])
        df.rename(columns={'Unnamed: 0': 'rows'}, inplace=True)
        dfs.append(df)

# normalize them all together
bigDf = pd.concat(dfs)
bigDfN, sdict, mdict = normalize(bigDf, cols)
# break them apart again
dfs = dict(tuple(bigDfN.groupby('gage')))

trainIAs = {}
trainDAs = {}
tvIAs = {}
tvDAs = {}
for df in dfs.values():
    gage = df['gage'][0]
    print(gage)
    df = df.reset_index()
    # add gage column for now
    cols = ['gage', 'PR_sum', 'PRCP_sum', 'VPD_mean', 'TMMN_mean', 'TMMX_mean', 'ET_mean', 'SRAD_mean', 'SWE_mean', 'Q_mean']
    train_ia, train_da, test_ia, test_da = dataPrep(df, 26, cols, 'Q_mean', date2split=pd.Timestamp(year=2021, day=1, month=1, tz='utc'))
    trainIAs[gage] = train_ia
    trainDAs[gage] = train_da
    tvIAs[gage] = test_ia
    tvDAs[gage] = test_da

trainIA = np.vstack(list(trainIAs.values()))
trainDA = np.hstack(list(trainDAs.values()))

# seperate the test data into test and validation
testing, validation = split_list_randomly(gages)# uncomment these two lines if you want to randomly make new training / validation

vIAs = {}
vDAs = {}
tIAs = {}
tDAs = {}
for gage in testing:
    tIAs[gage] = tvIAs[gage]
    tDAs[gage] = tvDAs[gage]

for gage in validation:
    vIAs[gage] = tvIAs[gage]
    vDAs[gage] = tvDAs[gage]

tIA = np.vstack(list(tIAs.values()))
tDA = np.hstack(list(tDAs.values()))
vIA = np.vstack(list(vIAs.values()))
vDA = np.hstack(list(vDAs.values()))

"""
Yay the data is prepped. Now run the model
"""
Timesizes = [13, 26, 32] #(half a year, 1 year, 15 months)
# train lstm on all the gages
model = runLSTM(trainIA, trainDA, 26, cols, 'sigmoid', 50)
# predict results using the validation data
predictions = model.predict(vIA)

# unnormalize it
pf = unNormalize_ar(predictions, mdict['Q_mean'], sdict['Q_mean'])

# calculate r squared
rsq = runRegression(xvars=vDA, y=predictions)
# calculate nse
nse = he.evaluator(he.nse, predictions, vDA)[0]

gages = []
nses = []
rsqs = []
for gage in vIAs.keys():
    predictions = model.predict(vIAs[gage])
    rsq = runRegression(xvars=vDAs[gage], y=predictions)
    nse = he.evaluator(he.nse, predictions, vDAs[gage])[0]
    gages.append(gage)
    nse.append(nse)
    rsqs.append(rsq)
    print(f'{gage}')



