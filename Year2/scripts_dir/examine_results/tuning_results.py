import pandas as pd
import os

results_dir = r'D:\Will_Git\Ozone_ML\Year2\results\aggregated_metrics\tuning'

results = []
for file in os.listdir(results_dir):
    path = os.path.join(results_dir, file)
    df = pd.read_csv(path)

    fileParams = file[17:-4]
    d = fileParams.find('d')
    dropout = fileParams[d+1:d+4]

    try:
        dropout=float(dropout)
    except:
        dropout=int(dropout[0])

    t = fileParams.find('t')
    timesize = int(fileParams[t+1:t+3])


    b = fileParams.find('b')
    batch = int(fileParams[b+1:b+3])

    u = fileParams.find('u')
    unit = fileParams[u+1:]

    rmss = []
    rsqs = []
    for col in df.columns:
        if 'rms' in col:
            rmss.append(df[col].mean())
        elif 'rsq' in col:
            rsqs.append(df[col].mean())

    mean_rms = sum(rmss) / len(rmss)
    mean_rsq = sum(rsqs) / len(rsqs)

    results.append({'dropout':dropout, 'units':unit, 'batch':batch, 'timesize':timesize, 'rsq':mean_rsq, 'rmse':mean_rms})

metrics = pd.DataFrame.from_dict(results)
sorted_metrics = metrics.sort_values(by='rsq', ascending=False)

sorted_metrics = metrics.sort_values(by='rmse', ascending=True)

