import os
import pandas as pd
import os
from pathlib import Path
import sys
baseDir = r'D:\Will_Git'
sys.path.append(r'{}\neuralhydrology'.format(baseDir))
from pathlib import Path
from neuralhydrology.evaluation import metrics, get_tester
from neuralhydrology.nh_run import start_run, eval_run
from neuralhydrology.utils.config import Config
import pickle
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

baseDir = 'D:\Will_Git'

# Collect the results of the ozone aware forecasts over each site for each forecast horizon

tuningResults = []
for run in os.listdir(fr'{baseDir}\Ozone_ML\forecast_runs'):
    if True:
        tuningR = {}
        print(run)
        horizon = run[0:2]
        print(f'horizon: {horizon}')
        run_dir = Path(rf"{baseDir}\Ozone_ML\forecast_runs\{run}")
        epoch = 25

        try:
            #eval_run(run_dir=run_dir, period="test")
            with open(run_dir / "test" / f"model_epoch0{epoch}" / "test_results.p", "rb") as fp:
                results = pickle.load(fp)
        except:
            print(f'no {run}')
            continue

        results.keys()

        sites = ['Evergreen', 'Idaho Springs', 'Five Points', 'Welby', 'Highlands Ranch', 'Rocky Flats', 'Boulder', 'Chatfield Reservoir', 'Sunnyside', 'East Plains', 'South Table']
        # init a df
        df = pd.DataFrame()
        qobs = results['Boulder']['1H']['xr']['o3_obs']
        df.index = qobs['date']
        tuningR['run'] =run
        #stop
        for site in sites:
            # extract observations and simulations
            qobs = results[site]['1H']['xr']['o3_obs']
            df[f'o3_obs_{site}_{horizon}'] = qobs

            qsim = results[site]['1H']['xr']['o3_sim']
            df[f'o3_sim_{site}_{horizon}'] = qsim

            values = metrics.calculate_all_metrics(qobs.isel(time_step=-1), qsim.isel(time_step=-1))
            tuningR[f'{site}_NSE'] = values['NSE']
            tuningR[f'{site}_RMSE'] = values['RMSE']
            tuningR[f'{site}_R2'] = values['Pearson-r']
            print(f"NSE: {values['NSE']}, RMSE: {values['RMSE']}, R2: {values['Pearson-r']}")

        tuningResults.append(tuningR)
        df.to_csv(rf'D:\Will_Git\Ozone_ML\Year2\nh_results\forecast_csvs\forecasts_{horizon}.csv')
            #
            # fig, ax = plt.subplots(figsize=(16,10))
            # ax.plot(qobs['date'], qobs)
            # ax.plot(qsim['date'], qsim)
            # ax.set_ylabel("o3")
results = pd.DataFrame.from_dict(tuningResults)

RMcols = []
R2cols = []
NScols = []
for col in results.columns:
    if 'RMSE' in col:
        RMcols.append(col)
    elif 'R2' in col:
        R2cols.append(col)
    elif 'NSE' in col:
        NScols.append(col)

results['rmse_mean'] = results[RMcols].mean(axis=1)
results['R2_mean'] = results[R2cols].mean(axis=1)
results['nse_mean'] = results[NScols].mean(axis=1)
results = results.sort_values(by=['rmse_mean'])

cols = list(results.columns)
cols.remove('run')

column_means = results[cols].mean()

# Create a new row containing the column means
results.loc['mean'] = column_means

#results.to_csv(r'D:\Will_Git\Ozone_ML\forecast_runs\1stGo.csv')

results

removeColumns = []
for col in results.columns:
    if 'NSE' in col or 'R2' in col:
        removeColumns.append(col)

rmse_results = results.drop(columns=removeColumns)
rmse_results = rmse_results.transpose()
rmse_results.to_csv(r'D:\Will_Git\Ozone_ML\Year2\nh_results\by_site_metrics\rmse.csv')

removeColumns = []
for col in results.columns:
    if 'RMSE' in col or 'NSE' in col:
        removeColumns.append(col)

rmse_results = results.drop(columns=removeColumns)
rmse_results = rmse_results.transpose()
rmse_results.to_csv(r'D:\Will_Git\Ozone_ML\Year2\nh_results\by_site_metrics\r2.csv')




resultsT = results.transpose()

# Collect the results of the ozone UNaware forecasts over each site

tuningResults = []
run = 'ua_1st_1503_165139'
tuningR = {}
print(run)
horizon = run[0:2]
print(f'horizon: {horizon}')
run_dir = Path(rf"{baseDir}\Ozone_ML\runs\{run}")
epoch = 25

try:
    #eval_run(run_dir=run_dir, period="test")
    with open(run_dir / "test" / f"model_epoch0{epoch}" / "test_results.p", "rb") as fp:
        results = pickle.load(fp)
except:
    print(f'no {run}')


# collect the accuracy results of the ozone unaware model over our sites
sites = ['Evergreen', 'Idaho Springs', 'Five Points', 'Welby', 'Highlands Ranch', 'Rocky Flats', 'Boulder', 'Chatfield Reservoir', 'Sunnyside', 'East Plains', 'South Table']
# init a df
df = pd.DataFrame()
qobs = results['Boulder']['1H']['xr']['o3_obs']
df.index = qobs['date']
tuningR['run'] =run
#stop
for site in sites:
    # extract simulations, we have no obs

    qsim = results[site]['1H']['xr']['o3_sim']
    df[f'o3_sim_{site}_{horizon}'] = qsim

    qobs = results[site]['1H']['xr']['o3_obs']
    df[f'o3_obs_{site}_{horizon}'] = qobs

    values = metrics.calculate_all_metrics(qobs.isel(time_step=-1), qsim.isel(time_step=-1))
    tuningR[f'{site}_NSE'] = values['NSE']
    tuningR[f'{site}_RMSE'] = values['RMSE']
    tuningR[f'{site}_R2'] = values['Pearson-r']
    print(f"NSE: {values['NSE']}, RMSE: {values['RMSE']}, R2: {values['Pearson-r']}")

tuningResults.append(tuningR)
df.to_csv(rf'D:\Will_Git\Ozone_ML\Year2\nh_results\forecast_csvs\forecasts_unaware_validation.csv')
    #
    # fig, ax = plt.subplots(figsize=(16,10))
    # ax.plot(qobs['date'], qobs)
    # ax.plot(qsim['date'], qsim)
    # ax.set_ylabel("o3")
results2 = pd.DataFrame.from_dict(tuningResults)

RMcols = []
R2cols = []
NScols = []
for col in results2.columns:
    if 'RMSE' in col:
        RMcols.append(col)
    elif 'R2' in col:
        R2cols.append(col)
    elif 'NSE' in col:
        NScols.append(col)

results2.to_csv(r'D:\Will_Git\Ozone_ML\forecast_runs\1stGo_ua.csv')

# collect the simulations that aren't over stations

# init a df
df = pd.DataFrame()
qobs = results['1']['1H']['xr']['o3_obs']
df.index = qobs['date']
tuningR['run'] =run
#stop

for i in range(1,141):
    # extract simulations, we have no obs
    i = str(i)
    qsim = results[i]['1H']['xr']['o3_sim']
    df[f'o3_sim_{i}'] = qsim

# save it
df.to_csv(r"D:\Will_Git\Ozone_ML\Year2\nh_results\forecast_csvs\unaware_forecasts.csv")