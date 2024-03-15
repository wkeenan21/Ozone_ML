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


df = pd.read_csv(r"D:\Will_Git\Ozone_ML\Year2\Merged_Data\merge4.csv")
df = df.dropna()
df.rename(columns={'sample_measurement':'o3'}, inplace=True)

dfs = dict(tuple(df.groupby('site_name')))

sites = ['Evergreen', 'Idaho Springs', 'Five Points', 'Welby', 'Highlands Ranch', 'Rocky Flats', 'Boulder', 'Chatfield Reservoir', 'Sunnyside', 'East Plains', 'South Table']
for site in sites:
    print(site)
    pp = PdfPages(r'D:\Will_Git\Ozone_ML\Year2\results\plots\site_by_site_correlations\{}.pdf'.format(site))
    for site2 in sites:
        df = pd.merge(dfs[site], dfs[site2], on='datetime')
        df = df.dropna()
        if site == site2:
            continue

        # transform to ppb
        for col in df.columns:
            if 'preds' in col or 'actual' in col:
                df[col] = df[col] * 1000
                # remove outliers
                df[col] = df[col].where(df[col] > 0, 0)
                df[col] = df[col].where(df[col] < 150, 150)

        fig1 = plotScat(df, 'o3_x', xaxisLabel=f'{site} Ozone (ppb)', yaxisLabel=f'{site2} Ozone (ppb)', yaxis='o3_y', title=f'{site} vs {site2} ozone')
        pp.savefig(fig1)
        plt.close()

    pp.close()
