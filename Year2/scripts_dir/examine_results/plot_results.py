import pandas as pd
import os
import datetime as dt
from matplotlib.backends.backend_pdf import PdfPages
from Year2.scripts_dir.from_usgs.wills_functions import *
from sklearn.metrics import mean_squared_error
from scipy.stats import gaussian_kde

def plotScat(df, xaxis, yaxis, yaxisLabel=None, filt = None, yaxis2 = None, xaxisLabel = None, yaxis2Label = None, title = None, legLabel1=None, legLabel2=None, font='Times New Roman', fontsize=15, linreg=True, logreg=False, expreg=False, annotate_shift=0, saveDir=False):
    fig, ax = plt.subplots()
    tsfont = {'fontname': font, 'size': fontsize} # gotta add this as a kwarg to change the font on things, except legend
    fontParams = {'family' : font,'size': fontsize}
    matplotlib.rc('font', **fontParams) # this changes the font of the legend for some reason
    # transform to ppb


    # make labels
    if not yaxisLabel:
        yaxisLabel = yaxis
    if not yaxis2Label:
        yaxis2Label = yaxis2

    # filter is if we want to create multiple lines based on unique values of a column
    if filt != None:
        for year in df[filt].unique():
            filter = df[filt] == year
            ax.scatter(df[filter][xaxis], df[filter][yaxis], label=year)
        ax.set_xlabel(xaxis, **tsfont)
        ax.legend(frameon=True)
    else:
        if not legLabel1:
            legLabel1 = yaxisLabel
            # Calculate the point density
            xy = np.vstack([df[xaxis], df[yaxis]])
            z = gaussian_kde(xy)(xy)
        ax.scatter(df[xaxis], df[yaxis], c=z, label=legLabel1)


    if yaxis2 != None:
        if not legLabel1:
            legLabel2 = yaxis2Label
        #ax2 = ax.twinx()
        ax.scatter(df[xaxis], df[yaxis2], label=legLabel2, color='green')
        ax.set_ylabel(yaxis2Label, **tsfont)

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
        ax.annotate('$R^2$ = ' + str(r_value**2)[0:4], (df[xaxis].max()- (df[xaxis].max()/10*3),df[yaxis].min()), **tsfont)
        rmse = mean_squared_error(df[xaxis], df[yaxis], squared=False)
        ax.annotate('RMSE = ' + str(rmse)[0:4], (df[xaxis].max() -(df[xaxis].max()/10),df[yaxis].min()), **tsfont)
        print(f'p value: {p_value}')

    ax.set_ylabel(yaxisLabel)
    ax.legend(loc='upper left', frameon=True)
    # make the title
    plt.title(title, **tsfont)
    fig = plt.gcf()

    fig.set_size_inches(12, 7)
    if saveDir:
        plt.savefig(os.path.join(saveDir, yaxis + '_vs_' + xaxis + '.png'))
    plt.show()
    return fig

sites = ['Evergreen', 'Idaho Springs', 'Five Points', 'Welby', 'Highlands Ranch', 'Rocky Flats', 'Boulder', 'Chatfield Reservoir', 'Sunnyside', 'East Plains', 'South Table']

testName = 'ozone_aware_universal_2024-03-13'
metricsList = []
for site in sites:
    metrics = {}
    print(site)
    metrics['site'] = site
    plotFold = fr'D:\Will_Git\Ozone_ML\Year2\results\plots\v3\{testName}'
    if not os.path.exists(plotFold):
        os.makedirs(plotFold)
    pp = PdfPages(r'{}\{}_24.pdf'.format(plotFold, site))
    res = r"D:\Will_Git\Ozone_ML\Year2\results\v3\{}\{}.csv".format(testName, site)
    df = pd.read_csv(res)
    df['date'] =pd.to_datetime(df['date'], utc=False)

    # transform to ppb
    for col in df.columns:
        if 'preds' in col or 'actual' in col:
            df[col] = df[col] * 1000
            # remove outliers
            df[col] = df[col].where(df[col] > 0, 0)
            df[col] = df[col].where(df[col] < 150, 150)

    for i in range(6):
        j = i+1
        rms = mean_squared_error(df['actual'], df[f'preds_{i}_{site}'], squared=False)
        model1 = LinearRegression()
        X = np.array(df['actual']).reshape(-1, 1)
        model1.fit(X, df[f'preds_{i}_{site}'])
        rsq = model1.score(X, df[f'preds_{i}_{site}'])
        metrics[f'{j} Hour RMSE'] = rms
        metrics[f'{j} Hour R2'] = rsq
    metricsList.append(metrics)


    fig1 = plotScat(df, 'actual', xaxisLabel='Observed Ozone (ppb)', yaxisLabel='1 Hour Forecasted Ozone (ppb)', yaxis=f'preds_0_{site}', title=site,)

    fig2 = plotScat(df, 'actual', xaxisLabel='Observed Ozone (ppb)', yaxisLabel='6 Hour Forecasted Ozone (ppb)', yaxis=f'preds_5_{site}', title=site,)

    fig3 = plotScat(df, 'actual', xaxisLabel='Observed Ozone (ppb)', yaxisLabel='8 Hour Forecasted Ozone (ppb)', yaxis=f'preds_7_{site}', title=site,)
    plt.close()
    plt.close()
    plt.close()
    pp.savefig(fig1)
    pp.savefig(fig2)
    pp.savefig(fig3)
    pp.close()

answer = pd.DataFrame.from_dict(metricsList)
answer.to_csv(r'{}\metrics.csv'.format(plotFold))

# plot estimation results
res_dir = r'D:\Will_Git\Ozone_ML\Year2\results\interp_error_testing\result_csvs'
dfs = []
for file in os.listdir(res_dir):
    df = pd.read_csv(os.path.join(res_dir, file))
    dfs.append(df)

bigDf = pd.concat(dfs)
for site in sites:
    print(site)
    pp = PdfPages(r'D:\Will_Git\Ozone_ML\Year2\results\plots\estimation_results\3\{}.pdf'.format(site))
    df = bigDf[bigDf['site_name'] == site]

    #df['date'] =pd.to_datetime(df['date'], utc=False)
    for i in range(6):
        rms = mean_squared_error(df['actual'], df[f'preds_{i}'], squared=False)
        model1 = LinearRegression()
        X = np.array(df['actual']).reshape(-1, 1)
        model1.fit(X, df[f'preds_{i}'])
        rsq = model1.score(X, df[f'preds_{i}'])
        j = i+1
        fig1 = plotScat(df, 'actual', xaxisLabel='Observed Ozone (ppb)', yaxisLabel=f'{j} Hour Forecasted Ozone (ppb)',
                        yaxis=f'preds_{i}', title=site)
        pp.savefig(fig1)
        plt.close()
    pp.close()



