import mne
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
import scipy

import matplotlib

font = {'family' : 'calibri', 'size'   : 9}
plt.rcParams['svg.fonttype'] = 'none'
matplotlib.rc('font', **font)

shapes = ['rectangle', 'square', 'isoTrapezoid', 'parallelogram', 'losange', 'kite', 'rightKite', 'rustedHinge', 'hinge', 'trapezoid', 'random']
subs = [f"sub-{(i+1):02}" for i in range(20)]
weights = 1 - np.array([0.06517094, 0.07478632, 0.15079365, 0.17826618, 0.21428571, 0.23000611, 0.25671551, 0.28469678, 0.30280830, 0.35724461, 0.42156085])
colors=["#DB8E00", "#F8766D", "#AEA200", "#64B200", "#00BD5C", "#00C1A7", "#00BADE", "#00A6FF", "#B385FF", "#EF67EB", "#FF63B6"]
times = np.arange(-.1,1.1,0.004)

def pretty_plot(ax=None):
    if ax is None:
        plt.gca()
    ax.tick_params(colors='dimgray')
    ax.xaxis.label.set_color('dimgray')
    ax.yaxis.label.set_color('dimgray')
    try:
        ax.zaxis.label.set_color('dimgray')
    except AttributeError:
        pass
    try:
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
    except ValueError:
        pass
    ax.spines['left'].set_color('dimgray')
    ax.spines['bottom'].set_color('dimgray')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    return ax

def vcorrcoef(X,y):
    Xm = np.mean(X,axis=0)
    ym = np.mean(y)
    r_num = np.sum((X-Xm)*(y-ym)[:,None],axis=0)
    r_den = np.sqrt(np.sum((X-Xm)**2,axis=0)*np.sum((y-ym)**2))
    r = r_num/r_den
    return r

conds = {"Outlier_Shared_Training": "+outlier+rocauc+perShape_decoding"}
name = "Outlier_Shared_Training"
path = "+outlier+rocauc+perShape_decoding"

all_decod_perf = np.zeros((20,11,301))
for idx, s in enumerate(subs):
    d = np.load(f"../bids_data/derivatives/msm/{s}/meg/{s}_task-POGS_proc-reference{path}.npy")
    if len(d.shape) == 2:
        all_decod_perf[idx, :, :] = d
    else:
        for ids in range(11):
            d[ids, ids, :, :] = np.nan
            for t in range(301):
                all_decod_perf[idx, ids, t] = np.nanmean(d[ids, :, t, t], axis=0)

imin = int(.10 * 250)
imax = int(imin + 250 * .8)

all_decod_perf = all_decod_perf[:,:,imin:imax]

scores = np.nan * np.zeros((20,11))
for sub in range(20):
    for i in range(11):
        try:
            scores[sub, i] = np.where(all_decod_perf[sub,i,:] > .57)[0][0] / 250
        except IndexError:
            pass

slopes = []
for i in range(20):
    mask = ~np.isnan(scores[i,:])
    slope, intercept, r, p, _ = scipy.stats.linregress(weights[mask],scores[i,:][mask])
    slopes.append(slope)

print(scipy.stats.ttest_1samp(slopes, popmean=0, alternative='less'))
