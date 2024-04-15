import os
from pathlib import Path
import sys
baseDir = r'D:\Will_Git'
sys.path.append(r'{}\neuralhydrology'.format(baseDir))
import matplotlib.pyplot as plt
import torch
from neuralhydrology.evaluation import metrics, get_tester
from neuralhydrology.nh_run import start_run, eval_run
from neuralhydrology.utils.config import Config
import pickle
import yaml
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd


tuningResults = []
for file in os.listdir(r'D:\Will_Git\Ozone_ML\variable_importance'):
    tuningR = {}
    run = file

    print(run)
    tuningR['horizon'] = run[0]

    if tuningR['horizon'] == 2:
        horizon = 24
    else:
        horizon = run[0]

    vars = [f'o3_{horizon}hour', 'u10', 'v10', 'r2', 'sp', 't2m', 'dswrf', 'MAXUVV', 'MAXDVV']
    for var in vars:
        if var in run:
            tuningR['missing_var'] = var
    run_dir = Path(rf"{baseDir}\Ozone_ML\variable_importance\{run}")
    epoch = 25


    try:
        eval_run(run_dir=run_dir, period="test")
        with open(run_dir / "test" / f"model_epoch0{epoch}" / "test_results.p", "rb") as fp:
            results = pickle.load(fp)
    except:
        print(f'no {run}')
        # eval_run(run_dir=run_dir, period="test")
        # with open(run_dir / "test" / f"model_epoch0{15}" / "test_results.p", "rb") as fp:
        #     results = pickle.load(fp)
        continue

    results.keys()

    sites = ['Evergreen', 'Idaho Springs', 'Five Points', 'Welby', 'Highlands Ranch', 'Rocky Flats', 'Boulder', 'Chatfield Reservoir', 'Sunnyside', 'East Plains', 'South Table']
    for site in sites:
        # extract observations and simulations
        qobs = results[site]['1H']['xr']['o3_obs']
        qsim = results[site]['1H']['xr']['o3_sim']
        #
        # fig, ax = plt.subplots(figsize=(16,10))
        # ax.plot(qobs['date'], qobs)
        # ax.plot(qsim['date'], qsim)
        # ax.set_ylabel("o3")
        # ax.set_title(f"Test period - NSE {results[site]['1H']['NSE']:.3f}")
        # plt.show()

        values = metrics.calculate_all_metrics(qobs.isel(time_step=-1), qsim.isel(time_step=-1))
        print(site)
        tuningR[f'{site}_NSE'] = values['NSE']
        tuningR[f'{site}_RMSE'] = values['RMSE']
        tuningR[f'{site}_R2'] = values['Pearson-r']
        print(f"NSE: {values['NSE']}, RMSE: {values['RMSE']}, R2: {values['Pearson-r']}")

        tuningResults.append(tuningR)

        # for key, val in values.items():
        #     print(f"{key}: {val:.3f}")
results = pd.DataFrame.from_dict(tuningResults)
results = results.drop_duplicates()

cols = []
for col in results.columns:
    if 'RMSE' in col:
        cols.append(col)

results['rmse_mean'] = results[cols].mean(axis=1)