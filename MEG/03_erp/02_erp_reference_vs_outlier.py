import mne
import pickle
import numpy as np

import matplotlib
from matplotlib import pyplot as plt
from matplotlib import rc
from matplotlib import cm

from mne.channels.layout import (_find_topomap_coords, find_layout, _pair_grad_sensors, _merge_ch_data)

from scipy.ndimage import uniform_filter1d
from scipy.signal import savgol_filter
from scipy.stats import linregress

matplotlib.rcParams['font.family'] = "sans-serif"
matplotlib.rcParams['font.sans-serif'] = "Calibri"
matplotlib.rcParams['text.usetex'] = False
matplotlib.rcParams['svg.fonttype'] = 'none'

def pretty_plot(ax=None):
    if ax is None:
        ax = plt.gca()
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

chtype = "grad"
cluster, cluster_pv = pickle.load(open(f"cl_{chtype}.pkl", "rb"))

for cl in np.where(cluster_pv<.05)[0]:
    print([cl, cluster_pv[cl]])

subs = [f"sub-{(i+1):02}" for i in range(20)]
all_refs = [mne.read_evokeds(f"../bids_data/derivatives/mne-bids-pipeline/{sub}/meg/{sub}_task-POGS_ave.fif", verbose=0)[1] for sub in subs]
all_outs = [mne.read_evokeds(f"../bids_data/derivatives/mne-bids-pipeline/{sub}/meg/{sub}_task-POGS_ave.fif", verbose=0)[2] for sub in subs]

for cl in np.where(cluster_pv<.05)[0]:
    cl_e = np.unique(cluster[cl][1])
    cl_t = np.unique(cluster[cl][0])
    print([np.min(cl_t), np.max(cl_t)])
    avg_ref_persub = np.array([np.average(all_refs[i].get_data(picks=chtype)[cl_e,:], axis=0) for i in range(len(subs))])
    avg_out_persub = np.array([np.average(all_outs[i].get_data(picks=chtype)[cl_e,:], axis=0) for i in range(len(subs))])
    mu_ref = np.mean(avg_ref_persub, axis=0)
    mu_out = np.mean(avg_out_persub, axis=0)
    se_ref = np.std(avg_ref_persub, axis=0) / np.sqrt(20)
    se_out = np.std(avg_out_persub, axis=0) / np.sqrt(20)
    times = np.arange(-.1,1.1,.004)
    lower = np.nan * times.copy()
    upper = np.nan * times.copy()
    for t in cl_t:
        lower[t] = np.min([mu_ref[t], mu_out[t]])
        upper[t] = np.max([mu_ref[t], mu_out[t]])
    plt.close('all')
    plt.figure(figsize=(5,2))
    plt.axhline(y=0, color='k', linestyle='-', linewidth=0.3)
    plt.axvline(x=0, color='k', linestyle='-', linewidth=0.3)
    plt.axvline(x=1, color='k', linestyle='-', linewidth=0.3)
    plt.plot(times, mu_ref, label="ref", color="k", linewidth=2)
    plt.plot(times, mu_out, label="out", color="0.5", linewidth=2)
    plt.fill_between(times, mu_ref-se_ref, mu_ref+se_ref, alpha=.1, color="k", linewidth=0)
    plt.fill_between(times, mu_out-se_out, mu_out+se_out, alpha=.1, color="k", linewidth=0)
    plt.fill_between(times, lower, upper, alpha=.35, color="k", linewidth=0)
    plt.legend()
    pretty_plot()
    plt.tight_layout()
    plt.savefig(f"figs/difference_{chtype}_cl-{cl}.svg")

all_delta_ave = [mne.read_evokeds(f"../bids_data/derivatives/mne-bids-pipeline/{sub}/meg/{sub}_task-POGS_ave.fif", verbose=0)[3] for sub in subs]

for cl in np.where(cluster_pv<.05)[0]:
    plt.close('all')
    cl_e = np.unique(cluster[cl][1])
    cl_t = np.unique(cluster[cl][0])
    imin, imax = np.min(cl_t), np.max(cl_t)
    tmin, tmax = (-.1 +.004*imin), (-.1 +.004*imax)
    tmin, tmax = np.round(tmin, 3), np.round(tmax, 3)
    all_topos = [all_delta_ave[i].get_data(picks=chtype) for i in range(len(subs))]
    mean_subject = np.mean(all_topos, axis=0)
    mit = np.min(cl_t)
    mat = np.max(cl_t)
    mean_to_plot = np.mean(mean_subject[:, mit:mat], axis=1)
    mask = np.array([i in cl_e for i in range(mean_subject.shape[0])]).reshape((-1,1))
    fig, ax = plt.subplots(1, 2, figsize=(7, 2), gridspec_kw={'width_ratios': [20, 1]})
    ref_ave = all_delta_ave[0].copy()
    ref_ave.pick_types(chtype).crop(0,.001)
    ref_ave._data = mean_to_plot.reshape((-1,1))
    ref_ave.plot_topomap([0], extrapolate="local", ch_type=chtype, sensors=False, axes=ax, show=False, mask=mask)
    fig.savefig(f"figs/topomaps_{chtype}_cl-{cl}_{tmin}_{tmax}.svg", dpi=300)

shapes = ['square', 'rectangle', 'isoTrapezoid', 'parallelogram', 'losange', 'kite', 'rightKite', 'rustedHinge', 'hinge', 'trapezoid', 'random']
cmap=["#F8766D", "#DB8E00", "#AEA200", "#64B200", "#00BD5C", "#00C1A7", "#00BADE", "#00A6FF", "#B385FF", "#EF67EB", "#FF63B6"]

cl_ids = np.where(cluster_pv<.05)[0]
print(cl_ids)

for cl_id in cl_ids:
    electrodes = np.unique(cluster[cl_id][1])
    all_delta = np.nan * np.zeros((20,11,301))
    for isub, sub in enumerate(subs):
        d = mne.read_epochs(f"../bids_data/derivatives/mne-bids-pipeline/{sub}/meg/{sub}_task-POGS_proc-clean_epo.fif", verbose=0).pick_types(chtype)
        for ishape, shape in enumerate(shapes):
            c = mne.combine_evoked([d[f"{shape}/reference"].average(), d[f"{shape}/outlier"].average()], weights=[1,-1])
            all_delta[isub, ishape, :] = np.average(c.get_data()[electrodes,:], axis=0)
    mu_per_shape = np.mean(all_delta, axis=0)
    plt.close('all')
    plt.figure(figsize=(5,2))
    plt.axhline(y=0, color='k', linestyle='-', linewidth=0.3)
    plt.axvline(x=0, color='k', linestyle='-', linewidth=0.3)
    plt.axvline(x=1, color='k', linestyle='-', linewidth=0.3)
    for i in range(11):
      plt.plot(times, savgol_filter(mu_per_shape[i,:], 37, 2), color=cmap[i], label=shapes[i], linewidth=1)
    pretty_plot()
    plt.tight_layout()
    plt.legend()
    plt.savefig(f"figs/each_shape_delta_{chtype}_cl-{cl_id}.svg", dpi=300)
    weights = 1 - np.array([0.07478632, 0.06517094, 0.15079365, 0.17826618, 0.21428571, 0.23000611, 0.25671551, 0.28469678, 0.30280830, 0.35724461, 0.42156085])
    all_rs = np.nan * np.zeros((20,301))
    for i in range(301):
        for j in range(20):
            all_rs[j,i] = np.corrcoef(weights, all_delta[j,:,i])[0,1]
    mu_r = np.mean(all_rs, axis=0)
    se_r = np.std(all_rs, axis=0) / np.sqrt(20)
    try:
        _,cl,pv,_ = mne.stats.permutation_cluster_1samp_test(all_rs, n_permutations=10000, tail=1)
        nonparamshade = np.nan * np.zeros((301))
        for ipv, lpv in enumerate(pv):
            if lpv < .05:
                nonparamshade[cl[ipv]] = mu_r[cl[ipv]]
        plt.close('all')
        plt.figure(figsize=(5,2))
        plt.axhline(y=0, color='k', linestyle='-', linewidth=.3)
        plt.axvline(x=0, color='k', linestyle='-', linewidth=.3)
        plt.axvline(x=1, color='k', linestyle='-', linewidth=.3)
        plt.plot(times, mu_r, color="k")
        plt.fill_between(times, mu_r-se_r, mu_r+se_r, color="k",  alpha=0.1, linewidth=0)
        plt.fill_between(times, np.zeros((301)), nonparamshade, color="k",  alpha=0.3, linewidth=0)
        pretty_plot()
        plt.tight_layout()
        plt.savefig(f"figs/correlation_and_cluster_{chtype}_cl-{cl_id}.svg")
        for cli in np.where(pv < .05)[0]:
            idxs = cl[cli][0]
            mean_per_shape = np.mean(all_delta[:,:,idxs], axis=2)
            mean_per_shape = mean_per_shape*(10**14)
            avg_points = np.mean(mean_per_shape, axis=0)
            se_points = np.std(mean_per_shape, axis=0) / np.sqrt(20)
            slope, intercept, r, p, _ = linregress(weights,avg_points)
            plt.close('all')
            plt.figure(figsize=(1.5, 2.5))
            for i in range(11):
                plt.scatter(weights[i], avg_points[i], c=cmap[i], label=shapes[i])
                plt.errorbar(weights[i], avg_points[i], yerr=se_points[i], ecolor=cmap[i])
            plt.annotate(f"r²={np.round(r,2)}; p={np.round(p,3)}", (0.1, avg_points[i]))
            plt.axline(xy1=(0, intercept), slope=slope, color="k")
            pretty_plot()
            plt.tight_layout()
            plt.savefig(f"figs/scatter_correlation_{chtype}_cl-{cl_id}_subcl-{cli}.svg")
    except IndexError:
        print("No (sub)-cluster to be found here")
    except ValueError:
        print("No (sub)-cluster to be found here")
