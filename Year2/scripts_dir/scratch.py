from herbie import FastHerbie
import requests
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# Gets the EPA ozone sites
#sites = pd.read_csv(r"C:\Users\willy\Documents\GitHub\Ozone_ML\Year2\BasicGIS\ozone_sites.csv")
toDrop = []

day = '2022-09-01'
hours = pd.date_range(start=day, periods=1, freq="1H", )
FH = FastHerbie(hours, model="hrrr")

t = FH.inventory(':MAXUVV:|:HPBL:|:MAXDVV:|:DSWRF:|:HGT:surface|:PRES:surface')
t