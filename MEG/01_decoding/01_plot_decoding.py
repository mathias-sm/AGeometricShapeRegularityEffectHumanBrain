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

conds = {"Outlier_Shared_Training": "+outlier+rocauc+perShape_decoding",
         "Outlier_Separate_Training": "+outlier+rocauc_decoding",
         "Outlier_Shared_Training_Nosmooth": "+outlier+rocauc+perShape+unsmooth_decoding",
         "Outlier_Separate_Training_Nosmooth": "+outlier+rocauc+unsmooth_decoding",
         "All_References": "_confusionPerTime+ovo"}

for name, path in conds.items():
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

    avg_dec = np.mean(all_decod_perf, axis=(0,1))
    where_max_dec = np.where(avg_dec == np.max(avg_dec))[0][0]
    print("")
    print(f"================ Dealing with {name} ================")
    print(f"## Max timeserie has index {where_max_dec}, time {np.round(-.10 + where_max_dec/250, 4)}ms which corresponds to a decoding perf of {np.round(avg_dec[where_max_dec], 3)})")

    to_vcorr = np.swapaxes(all_decod_perf, 0, 1).reshape((11,-1))
    rs = vcorrcoef(to_vcorr, weights).reshape((20,301))
    _, clusters, cluster_pv, _ = mne.stats.permutation_cluster_1samp_test(rs, n_permutations=2**13, tail=1)
    # _, clusters, cluster_pv, _ = mne.stats.permutation_cluster_1samp_test(rs, n_permutations=2**13, tail=1, threshold=dict(start=0,step=0.1))

    fig, axd = plt.subplot_mosaic([['a', 'b', 'c', 'c']],
                                  figsize=(6, 2),
                                  constrained_layout=True)

    axd["b"].axhline(y=0, color='k', linestyle='-', linewidth=.3)
    axd["b"].axvline(x=0, color='k', linestyle='-', linewidth=.3)
    axd["b"].axvline(x=1, color='k', linestyle='-', linewidth=.3)
    axd["b"].set_xticks(ticks=[0,1])
    avg_r = np.mean(rs, axis=0)
    where_max_r = np.where(avg_r == np.max(avg_r))[0][0]
    print(f"## Max timeserie has index {where_max_r}, time {np.round(-.10 + where_max_r/250, 4)}ms which corresponds to {np.round(avg_r[where_max_r], 3)})")

    se_r = np.std(rs, axis=0)/np.sqrt(20)
    nice_fill = np.zeros((301))
    for cl in np.where(cluster_pv < .05)[0]:
        nice_fill[clusters[cl]] = avg_r[clusters[cl]]
        timings = -.1 + clusters[cl][0]/250
        print(f"## Cluster from {np.round(np.min(timings),4)}ms to {np.round(np.max(timings), 4)}ms")

    axd["b"].fill_between(times, np.zeros((301)), nice_fill, facecolor="k", alpha=.3)
    axd["b"].plot(times, avg_r, color="k")
    axd["b"].fill_between(times, avg_r-se_r, avg_r+se_r, facecolor="k", alpha=0.1)

    avg = np.mean(all_decod_perf, axis=0)
    se = np.std(all_decod_perf, axis=0)/np.sqrt(19)
    axd["a"].axhline(y=.5, color='k', linestyle='-', linewidth=.3)
    axd["a"].axvline(x=0, color='k', linestyle='-', linewidth=.3)
    axd["a"].axvline(x=1, color='k', linestyle='-', linewidth=.3)
    axd["a"].set_xticks(ticks=[0,1])
    for i in [1,0,2,3,4,5,6,7,8,9,10]:
        axd["a"].plot(times, avg[i,:], label=shapes[i], color=colors[i])
        axd["a"].fill_between(times, avg[i,:]-se[i,:], avg[i,:]+se[i,:], facecolor=colors[i], alpha=.1)

    mus_ts = np.nan*np.ones((20, 11, 301))
    mus = np.nan*np.ones((20, 11))
    for i in range(11):
        for cl in np.where(cluster_pv < .05)[0]:
            mus_ts[:, i, clusters[cl][0]] = all_decod_perf[:, i, clusters[cl][0]]
        mus[:,i] = np.nanmean(mus_ts[:,i,:], axis=1)

    mumu = np.mean(mus, axis=0)
    sdmu = np.std(mus, axis=0)/np.sqrt(20)
    slope, intercept, r, p, _ = scipy.stats.linregress(weights,mumu)
    for i in [1,0,2,3,4,5,6,7,8,9,10]:
        axd["c"].scatter(weights[i], mumu[i], c=colors[i], label=shapes[i])
        axd["c"].errorbar(weights[i], mumu[i], yerr=sdmu[i], ecolor=colors[i])

    r = str(round(r, 2))
    p = str(round(p, 4))
    axd["c"].annotate("r²=" + r + "; p=" + p, (0.6, mumu[i]))
    axd["c"].axline(xy1=(min(weights), intercept + min(weights)*slope), slope=slope, color="k")

    box = axd["c"].get_position()
    axd["c"].set_position([box.x0 + 0.1, box.y0, box.width * 0.5, 1.1*box.height])
    axd["c"].legend(loc='center left', bbox_to_anchor=(1, 0.5))

    for pl in ["a", "b", "c"]:
        axd[pl] = pretty_plot(axd[pl])

    fig.savefig(f"./figs_behavior/{name}.svg")
    fig.savefig(f"./figs_behavior/{name}.png", dpi=600)

