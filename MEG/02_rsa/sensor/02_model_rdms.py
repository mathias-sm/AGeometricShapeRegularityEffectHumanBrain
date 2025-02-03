import mne
import numpy as np
import rsatoolbox
import pandas as pd
import copy
import scipy.stats
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import matplotlib


font = {'family' : 'calibri', 'size'   : 9}
plt.rcParams['svg.fonttype'] = 'none'
matplotlib.rc('font', **font)


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


color = ["#f1a340", "#998ec3"]
subs = [f"sub-{(i+1):02}" for i in range(20)]

ordered = ["square", "rectangle", "isoTrapezoid", "parallelogram", "losange", "kite", "rightKite", "rustedHinge", "hinge", "trapezoid", "random"]

cm = "crossnobis"
for smooth in ["smooth", "unsmooth"]:
    rdms_movie = rsatoolbox.rdm.rdms.load_rdm(f"./all_rdms/rdms_{cm}_{smooth}.pkl")
    times = rdms_movie.subset('subj', ['sub-01']).rdm_descriptors['time']

    def parse_file(fname):
        """
        Parse an RDM csv file. I reorder things systematically to ensure that
        things end up in the same order across different RDMs.
        """
        diss_content = pd.read_csv(fname, index_col=0)
        diss_content = diss_content.reindex(index=ordered, columns=ordered)
        diss = diss_content.values
        tri_diss = diss[np.triu(np.ones_like(diss, dtype=bool), k=1)]
        rdm = rsatoolbox.rdm.RDMs(
                scipy.stats.zscore(tri_diss),
                pattern_descriptors={"shape": ordered})
        rdm.sort_by(shape=ordered)
        return rdm

    symbolic = parse_file("../../../derive_theoretical_RDMs/symbolic/symbolic_sym_diss_mat.csv")
    IT = parse_file("../../../derive_theoretical_RDMs/CNN/output/diss_mat_model-cornet_s_layer-IT.csv")

    model_rdms = copy.deepcopy(symbolic)
    model_rdms.append(copy.deepcopy(IT))
    model_rdms.rdm_descriptors['model_names'] = model_names = ['symbolic', 'IT']
    model = rsatoolbox.model.ModelWeighted('full', model_rdms)

    method = "corr_cov"

    fitted = np.zeros((20, 2, 301))
    for idxs, sub in enumerate(subs):
        ds = rdms_movie.subset('subj', sub)
        for idxt, t in enumerate(times):
            topred = ds.subset('time', [t])
            topred.dissimilarities[0] = scipy.stats.zscore(topred.dissimilarities[0])
            for im, m in enumerate([symbolic, IT]):
                fitted[idxs, im, idxt] = rsatoolbox.rdm.compare(m, topred, method=method)

    plt.close('all')
    plt.figure(figsize=(5, 2))
    plt.axhline(y=0, color='dimgray', linestyle='-', linewidth=0.5)
    plt.axvline(x=0, color='dimgray', linestyle='-', linewidth=0.5)
    plt.axvline(x=1, color='dimgray', linestyle='-', linewidth=0.5)
    plt.axvline(x=0.8, linewidth=0.5, linestyle=(0, (5, 10)), color='k')
    for i in range(1,-1,-1):
        r_ = fitted[:, i, :]
        mu = np.mean(r_, axis=0)
        se = np.std(r_, axis=0)/np.sqrt(20)
        tmin, tmax = 0, .8
        imin, imax = int((tmin + .1)/.004), int((tmax + .1)/.004)
        offset = .1/.004
        r_for_clust = r_[:,imin:imax+1]
        _, clusters, p_values, _ = mne.stats.permutation_cluster_1samp_test(
                r_for_clust,
                n_permutations=2**13,
                verbose=False,
                tail=1)
        p05 = np.nan * np.zeros((301))
        for x in np.where(p_values < .05)[0]:
            filt = [int(x) for x in clusters[x][0]+offset]
            p05[filt] = mu[filt]
            print(f"Model {model_names[i]}; from {np.min((np.array(filt)-offset) * .004)*1000}ms to {np.max((np.array(filt)-offset) * .004)*1000}ms")
        p1 = np.nan * np.zeros((301))
        plt.plot(times, mu, color=color[i], linewidth=0.4)
        plt.plot(times, p05, label=model_names[i], color=color[i])
        plt.fill_between(times, p05, np.zeros((301)), alpha=.5, facecolor=color[i])

    plt.xlabel('time')
    plt.ylabel(f'model-data crossnobis similarity')
    plt.legend()
    pretty_plot()
    plt.savefig(f"./figs/lm_{cm}_{method}_{smooth}.svg")
