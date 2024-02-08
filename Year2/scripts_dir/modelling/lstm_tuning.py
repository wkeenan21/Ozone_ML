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
import sys
sys.path.append(r"/Year2/scripts_dir")
# import a bajillion functions from my other script
from lstm_functions import *

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
        #print(input_sequence)
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
            print(band)
            #return band

    if len(bad_dates) == 0:
        print('no bad dates found')
    bad_times = boolean_array_from_3d_with_false(ia, bad_dates)
    return bad_times

#test = time_continuity_test(st_vIA, 10, 11, rdict, mdict)
# Import data
O3J = pd.read_csv(r"/Year2/Merged_Data/merge3.csv")
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

# decide your timesize
timesizes = [12, 24, 48]
for timesize in timesizes:
    new_dfs = []
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

    # # split the data by training, val, and testing
    # trainIAs_f_24, trainDAs_f_24, trainDates_f_24, vIAs_f_24, vDAs_f_24, vDates_f_24, tIAs_f_24, tDAs_f_24, tDates_f_24, dfs = split_data(O3Jn, sets, cols, False, 24)
    #
    # # stack them into big arrays for training the universal model
    # trainIA_f_24 = np.vstack(list(trainIAs_f_24.values()))
    # trainDA_f_24 = np.hstack(list(trainDAs_f_24.values()))
    # vIA_f_24 = np.vstack(list(vIAs_f_24.values()))
    # vDA_f_24 = np.hstack(list(vDAs_f_24.values()))
    # tIA_f_24 = np.vstack(list(tIAs_f_24.values()))
    # tDA_f_24 = np.hstack(list(tDAs_f_24.values()))

    trainIAs_t_24, trainDAs_t_24, trainDates_t_24, vIAs_t_24, vDAs_t_24, vDates_t_24, tIAs_t_24, tDAs_t_24, tDates_t_24, dfs2 = split_data(O3Jn, sets, cols, True, 24)
    trainIA_t_24 = np.vstack(list(trainIAs_t_24.values()))
    trainDA_t_24 = np.hstack(list(trainDAs_t_24.values()))
    vIA_t_24 = np.vstack(list(vIAs_t_24.values()))
    vDA_t_24 = np.hstack(list(vDAs_t_24.values()))
    tIA_t_24 = np.vstack(list(tIAs_t_24.values()))
    tDA_t_24 = np.hstack(list(tDAs_t_24.values()))
    #ia, da, nans = split_data(sets, cols, False, 24)

    #model = runLSTM(trainIA, trainDA, 72, cols=cols, activation='sigmoid', epochs=25, units=32, run_model=True)
    epochs = [25]
    batches = [32, 64, 128]
    units = [32, 64, 128]
    drops = [0, 0.2, 0.4]
    activation = []
    for e in epochs:
        for b in batches:
            for u in units:
                for d in drops:
                    model_one_hot = trainLSTMgpt(trainIA_t_24, trainDA_t_24, epochs=e, batch=b, units=u, drop=d)
                    #model_no_one_hot = trainLSTMgpt(trainIA_f_24, trainDA_f_24, epochs=25)


                    # # drop ozone from training data and see how it does
                    # trainIA_noO3 = np.delete(trainIA, 0, axis=2)
                    # vIA_noO3 = np.delete(vIA, 0, axis=2)

                    # see how it did by site
                    merged_dfs = []
                    metricsList = []
                    rm = pd.DataFrame()
                    for site in vIAs_t_24.keys():
                        print(site)
                        lat = dfs[site]['latitude'].max()
                        lon = dfs[site]['longitude'].max()
                        st_vIA = vIAs_t_24[site].copy()
                        st_vDA = vDAs_t_24[site].copy()
                        st_vDates = vDates_t_24[site].copy()

                        # multistep test
                        metrics = {}
                        metrics['site'] = site
                        preds, actuals, dates, rms_d, rsq_d = multistep_forecast(model_one_hot, 6, st_vIA, st_vDA, st_vDates, 0, rdict, mdict)
                        for i in rms_d.keys():
                            metrics[i] = rms_d[i]
                        for j in rsq_d.keys():
                            metrics[j] = rsq_d[j]
                        metricsList.append(metrics)

                        results = {}
                        for i in range(len(preds)):
                            results[i] = pd.DataFrame()
                            results[i][f'date'] = dates[i]
                            uP = unNormalize(preds[i], rdict['o3'], mdict['o3'])
                            results[i][f'preds_{i}_{site}'] = uP.flatten()
                        # add the actual o3 once
                        results[0]['actual'] = unNormalize(actuals[0], rdict['o3'], mdict['o3'])

                        from functools import reduce
                        # Use functools.reduce and pd.merge to merge DataFrames on the 'date' column
                        merged_df = reduce(lambda left, right: pd.merge(left, right, on='date'), results.values())
                        merged_df['site_name'] = site
                        merged_df['lat'] = lat
                        merged_df['lon'] = lon
                        merged_dfs.append(merged_df)
                        #merged_df.to_csv(r"D:\Will_Git\Ozone_ML\Year2\results\universal_one_hot\{}_6hour_24time_n.csv".format(site))

                    df = pd.DataFrame(metricsList)
                    df.to_csv(r"D:\Will_Git\Ozone_ML\Year2\results\aggregated_metrics\tuning\universal_tuning_t{}_e{}_d{}_b{}_u{}.csv".format(timesize, e, d, b, u))

                    #merged_df = reduce(lambda left, right: pd.merge(left, right, on='date'), merged_dfs)


