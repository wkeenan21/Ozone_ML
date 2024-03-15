from pathlib import Path
import sys
baseDir = r'C:\Users\wkeenan\OneDrive - DOI\Documents\GitHub'
sys.path.append(r'{}\neuralhydrology'.format(baseDir))
import matplotlib.pyplot as plt
import torch
from neuralhydrology.evaluation import metrics, get_tester
from neuralhydrology.nh_run import start_run, eval_run
from neuralhydrology.utils.config import Config
import pickle


config_path = Path(rf"{baseDir}\Ozone_ML\Year2\scripts_dir\neuralHydro\config.yml")
run_config = Config(config_path)
# print('model:\t\t', run_config.model)
# print('use_frequencies:', run_config.use_frequencies)
# print('seq_length:\t', run_config.seq_length)
# print('dynamic_inputs:', run_config.dynamic_inputs)
runName = run_config.experiment_name
print(runName)

start_run(config_file=config_path, gpu=-1)

run_dir = Path(rf"C:\Users\wkeenan\OneDrive - DOI\Documents\GitHub\Ozone_ML\runs\test_run_1303_230248")
eval_run(run_dir=run_dir, period="test")

with open(run_dir / "test" / "model_epoch050" / "test_results.p", "rb") as fp:
    results = pickle.load(fp)

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
    print(f"NSE: {values['NSE']}, RMSE: {values['RMSE']}, R2: {values['Pearson-r']}")
    # for key, val in values.items():
    #     print(f"{key}: {val:.3f}")