import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib
from datetime import timedelta
import itertools
import pandas as pd
import warnings
import Utils 
plt.ion()
import seaborn as sns
sns.set()

####################
### NEEDS REWRITE ###
#####################
def plot_results(results, rolling=24, groupby=None):
    """ Plots the results after prediction ('predict_load' with graph=true). Either complete time series or boxplot if aggregated errors """
    if groupby is None:
        rolling_col = "{0} mean".format(rolling)
        fig, axes = plt.subplots(nrows=2, ncols=1)
        results[rolling_col] = results['Error (%)'].rolling(window=rolling, center=False).mean()
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1]) 
        ax = results.plot(y=['Prediction', 'True'], ax=plt.subplot(gs[0]), sharex=True)
        ax2 = results.plot(y=['Error (%)', rolling_col], ax=plt.subplot(gs[1]), sharex=True)
        ax.set_title("Load prediction plot", fontsize=13)
        ax2.set_title("Forecasting error (%)".format(groupby), fontsize=13)

        ax.set_ylabel("Electricity load (MW)", fontsize=11)
        ax2.set_ylabel("Error (%)")

        ax2.set_xlabel("Time")

    else:
        grp_by, labels = Utils.get_groupby_time(results, groupby)
        ax = results.boxplot(by=grp_by, column='Error (%)', showfliers=False, rot=90, fontsize=11)
        if labels is not None:
            locs, _ = plt.xticks()
            plt.xticks(locs, labels)

        plt.suptitle("")    
        ax.set_title("Forecasting error (%) per {0}".format(groupby), fontsize=13)
        ax.set_xlabel(groupby, fontsize=11)
        ax.set_ylabel("Forecasting error (%)", fontsize=11)

    # Support for plotting outside a Jupyter notebook
    backend = matplotlib.get_backend()
    if "backend_inline" not in backend:
        plt.show(block=True)


def show_save(action, fname=None):
    """ Used to not yet show a graph that is constructed in multiple steps """
    if fname is not None and action == "SAVE":
        plt.legend()
        plt.savefig(fname + ".png")
        plt.clf()
    elif action == "SHOW":
        plt.legend()
    else:
        if action == "SAVE" and fname is None:
            warnings.warn("Save without file name !")
        else:
            warnings.warn("Unknown plot action {0}, not showing nor saving".format(action))


def plot_load(data, action="SHOW", labels=[""]):
    """ load vs time plot """ 
    if not isinstance(data, list):
        data = [data]

    ax = data[0].plot.line(x="Period", y="TS", label=labels[0])
    
    if len(data) > 1:
        for d, l in itertools.zip_longest(data[1:], labels[1:]):
            d.plot.line(x="Period", y="TS", label=l, ax=ax)

    show_save(action)


#=========================================================================
### Barely usedmethods
### Would be changed if project continues - not worth talking much about it 
#========================================================================== 
def plot_diff_loc(data, diff, labels=[], action="SHOW", fname=None):
    plot_load(data, labels=labels, action="NOTHING")
    """ Plots the load with the corrected error locations """
    plt.scatter(list(diff['Period']), list(diff['TS']), s=50, label="Anomalie(s)", marker='x', c='red', zorder=10)
    
    plt.ylabel("Series value")
    plt.xlabel("Series time")
    show_save(action, fname=fname)

def plot_all_anomalies(uncorrected, corrected, diff, action="SHOW"):
    for x in range(len(diff)):
        plot_anomaly(uncorrected, corrected, diff, x, action=action)

def plot_anomaly(uncorrected, corrected, anomaly, action="SHOW"):
    # Could have picked middle anomaly.. 
    anomaly_date = list(anomaly['Period'])[0]
    start = anomaly_date - timedelta(days=1)
    end = anomaly_date + timedelta(days=1)
    anomaly_period_chunk = (corrected['Period'] >= start) & (corrected['Period'] <= end)
    
    ncor = uncorrected.loc[anomaly_period_chunk]
    cor  = corrected.loc[anomaly_period_chunk]
    plot_diff_loc([ncor, cor], anomaly, labels=['Uncorrected', 'Corrected'], action=action, fname="Anomaly_"+str(anomaly_date))
    
def plot_around(period, data, title=""):
    """ Used to plot a specific error and its surrounding values 
        Seems similar to plot_anomaly, cannot remember the difference now 
    """
    start = period - timedelta(days=1)
    end = period + timedelta(days=1)
    
    period_chunk = (data['Period'] >= start) & (data['Period'] <= end)
    anomaly_chunk = pd.DataFrame(data.loc[period_chunk])
    plot_load([anomaly_chunk], action="NOTHING")
    anomaly = anomaly_chunk.loc[anomaly_chunk['Period'] == period]

    plt.scatter(list(anomaly['Period']), list(anomaly['TS']), s=50, label="Anomalie(s)", marker='x', c='red', zorder=10)
    plt.title(title)

    show_save("SHOW")