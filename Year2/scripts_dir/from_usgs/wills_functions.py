import os
import matplotlib
try:
    matplotlib.use('TkAgg')
except:
    print('no TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

try:
    from tensorflow.keras.layers import Input, LSTM,Dense, Dropout
    from tensorflow.keras import layers, optimizers, losses, metrics, Model
    import tensorflow as tf
    tf.keras.callbacks.TerminateOnNaN()
except:
    print('no tensorflow')

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from math import log10, floor, log
import scipy.stats
import datetime as dt
from datetime import timedelta
import random
import hydroeval as he


import matplotlib.dates as md
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
    ax.legend(loc='upper left', frameon=True)
    # make the title
    plt.title(title, **tsfont)
    fig = plt.gcf()
    fig.set_size_inches(12, 7)
    if saveDir:
        plt.savefig(os.path.join(saveDir, yaxis + '_vs_' + xaxis + '.png'))
    plt.show()

def plotLines(df, xaxis, yaxis, yaxisLabel = None, filt = None, yaxis2 = None, yaxis2Label = None, yaxis3=None, yaxis3Label=None, changeYlim=None, title = None, linreg=False, dateSpacing=3, logY=False,annotate_shift=0, sameY=False, saveDir=None):
    df = df.dropna() # gotta drop na to get a regression line
    fig, ax = plt.subplots()
    if yaxis2Label == None:
        yaxis2Label = yaxis2

    if yaxisLabel == None:
        yaxisLabel = yaxis

    if yaxis3Label == None:
        yaxis3Label = yaxis3

    if filt != None:
        for year in df[filt].unique():
            filter = df[filt] == year
            ax.plot(df[filter][xaxis], df[filter][yaxis])
    else:
        if not sameY:
            ax.plot(df[xaxis], df[yaxis], label=yaxisLabel, linestyle='solid', marker='o')
            ax.legend(loc='upper left', frameon=True)
        elif sameY:
            ax.plot(df[xaxis], df[yaxis], label=yaxisLabel, linestyle='solid', marker='o')
            ax.plot(df[xaxis], df[yaxis2], label=yaxis2Label, color='green', linestyle='dashed')
            if yaxis3:
                ax.plot(df[xaxis], df[yaxis3], label=yaxis3Label, color='purple', linestyle='--')
            ax.legend(loc='upper left', frameon=True)

    if not sameY:
        axs = [ax]
        ys = [yaxis]
        if yaxis2 != None:
            ax2 = ax.twinx()
            ax2.plot(df[xaxis], df[yaxis2], label=yaxis2Label, color='green', linestyle='dashed', marker='o')
            ax2.set_ylabel(yaxis2Label)
            ax2.legend(loc='upper right', frameon=True)
            axs.append(ax2)
            ys.append(yaxis2)

        if yaxis3 != None:
            ax3 = ax.twinx()
            ax3.plot(df[xaxis], df[yaxis3], label=yaxis3Label, color='purple', linestyle='--', marker='o')
            #ax3.set_ylabel(yaxis3Label)
            leg = ax3.legend(loc='upper right', frameon=True)
            # yaxis2 and 3 need to be in the same units, in the same range
            axs.append(ax3)
            ys.append(yaxis3)

        if sameY:
            maxes = []
            for y in ys:
                maxes.append(df[y].max())
            m = max(maxes)
            for axe in axs:
                axe.set_ylim(0, m)

    if logY:
        plt.yscale('log')

    if changeYlim:
        for axe in axs:
            axe.set_ylim(0, changeYlim)

    ax.xaxis.set_major_locator(md.DayLocator(interval = dateSpacing))
    ax.xaxis.set_major_formatter(md.DateFormatter('%m/%Y'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation = 90)

    ax.set_xlabel(xaxis)
    ax.set_ylabel(yaxisLabel)

    fig = plt.gcf()
    fig.set_size_inches(14, 7)

    plt.title(title)

    if linreg:
        #m, b = np.polyfit(df[yaxis], df[yaxis2], 1)
        #plt.plot(df[yaxis], m * df[yaxis2] + b, color='black')
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(df[yaxis], df[yaxis2])
        if p_value < 0.001:
            p_value = '<0.01'
        else:
            p_value = str(round_to_1(p_value))
        ax.annotate('$R^2$ = ' + str(r_value**2)[0:4], (md.date2num(dt.datetime(2017, 5, 1)) +annotate_shift,df[yaxis].median()))

    if saveDir:
        plt.savefig(os.path.join(saveDir, '{}_{}_vs_{}.png'.format(gage, xaxis, yaxis)))

    return fig


# this function normalizes all the variables to be z scores
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

    return df2, mDict, sDict

def normalize(df, cols):
    df2 = df.copy()
    maxDict = {}
    minDict = {}
    for col in cols:
        max = df2[col].max()
        min = df2[col].min()
        df2[col] = (df2[col] - min) / (max - min)
        maxDict[col] = max
        minDict[col] = min

    return df2, maxDict, minDict

def unNormalizeZ_df(df, cols, mean, std):
    # unnormalizes a df on the columns you give it
    df2 = df.copy()
    for i in range(len(cols)):
        df2[cols[i]] = (df2[cols[i]] * std) + mean
    return df2

def unNormalizeZ_ar(ar, mean, std):
    # unnormalize an array
    ar2 = (ar * std) + mean
    return ar2

def unNormalize_ar(ar, max, min):
    # unnormalize an array
    ar2 = (ar * (max-min)) + min
    return ar2

def split_list(lst):
    # splits a list by selecting every other item and returning two lists
    selected_items = lst[::2]
    remaining_items = lst[1::2]
    return selected_items, remaining_items

def export_to_excel(dataframes_dict, excel_file_path):
    """
    Export a dictionary of pandas DataFrames to an Excel file.

    Parameters:
    - dataframes_dict (dict): A dictionary where keys are sheet names and values are pandas DataFrames.
    - excel_file_path (str): The path to the Excel file to be created.

    Returns:
    - None
    """
    with pd.ExcelWriter(excel_file_path, engine='xlsxwriter') as writer:
        for sheet_name, dataframe in dataframes_dict.items():
            dataframe.to_excel(writer, sheet_name=sheet_name, index=False)

def runRegression(X, y):
    # Ensure X is a 2D array with a single column
    X = X.reshape(-1, 1) if len(X.shape) == 1 else X
    # Create a linear regression model
    model = LinearRegression()
    # Fit the model
    model.fit(X, y)
    # Make predictions
    y_pred = model.predict(X)
    # Calculate R-squared
    r_squared = r2_score(y, y_pred)
    return r_squared

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
def runLSTM(ind_arr, dep_arr, timesize, activation, epochs, units=10, batch_size=32, learning_rate=0.001):
    '''configure the model'''
    input_var_cnt = ind_arr.shape[2] ##the number of variables used to perform prediction
    input_lstm = Input(shape=(timesize, input_var_cnt)) ##what is the input for every sample, sample count is not included every sample should be a 2D matrix
    ##prepare a LSTM layer
    lstmlayer = LSTM(units, activation=activation) (input_lstm) ##this outputs a matrix of 1*unit_lstm, the format is the layer (input), the output of the layer stores the time series info and the interaction of variables..
    denselayer = Dense(1)(lstmlayer) ## reduce the hidden dimension to 1 ==== output data ,1 value for 1 input sample
    model = Model(inputs = input_lstm, outputs = denselayer)
    #model.add(Dropout(0.2, input_shape=(60,)))
    #model.add(Dropout(dropout)) # 20% of the neurons get ignored at random times to prevent overfitting
    opt = optimizers.Adam(learning_rate=learning_rate, clipnorm=1) # set initial learning rate and clipnorm to prevent exploding gradients clipnorm=1
    model.compile(loss='mse', optimizer=opt) ##how to measure the accuracy  compute mean squared error using all y_pred, y_true

    model.fit(ind_arr, dep_arr, epochs=epochs, batch_size=32, verbose=2)
    return model