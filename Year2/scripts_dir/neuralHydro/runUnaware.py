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

for i in range(1, 141):
    print(i)

baseDir = r'D:\Will_Git'
config_path = Path(rf"{baseDir}\Ozone_ML\Year2\scripts_dir\neuralHydro\unaware_config.yml")
start_run(config_file=config_path, gpu=-1)