import pandas as pd
import os
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import numpy as np

res_dir = r'D:\Will_Git\Ozone_ML\Year2\results\interp_error_testing\result_csvs'
dfs = []
for file in os.listdir(res_dir):
    df = pd.read_csv(os.path.join(res_dir, file))
    dfs.append(df)

bigDf = pd.concat(dfs)
sites = ['Evergreen', 'Idaho Springs', 'Five Points', 'Welby', 'Highlands Ranch', 'Rocky Flats', 'Boulder', 'Chatfield Reservoir', 'Sunnyside', 'East Plains', 'South Table']

metricList = []
for i in range(6):
    for site in sites:
        df = bigDf[bigDf['site_name'] == site]
        rms = mean_squared_error(df['actual'], df[f'preds_{i}'], squared=False)
        model1 = LinearRegression()
        X = np.array(df['actual']).reshape(-1, 1)
        model1.fit(X, df[f'preds_{i}'])
        rsq = model1.score(X, df[f'preds_{i}'])
        print(i, site, rms, rsq)
        metrics = {'site_name':site, 'Forecast Horizon': i+1, 'RMSE': rms, 'R2': rsq}
        metricList.append(metrics)

metricDf = pd.DataFrame.from_dict(metricList)


grouped = metricDf.groupby('site_name').mean()

# Plotting
categories = grouped.index
bar_width = 0.2
index = range(len(categories))

# Plotting each numerical column for each category
for i, col in enumerate(grouped.columns):
    plt.bar([x + i * bar_width for x in index], grouped[col], width=bar_width, label=col)

plt.xlabel('Category')
plt.ylabel('Mean Value')
plt.title('Mean Values of Numerical Columns by Category')
plt.xticks([x + bar_width for x in index], categories)
plt.legend()
plt.show()