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


subs = [f"sub-{(i+1):02}" for i in range(20)]
cm = "crossnobis"
method = "corr_cov"
ordered = ["square", "rectangle", "isoTrapezoid", "parallelogram", "losange", "kite", "rightKite", "rustedHinge", "hinge", "trapezoid", "random"]

# DINO Layers
#models = {f"DINOv2_l{k}": f"../../../derive_theoretical_RDMs/more_NNs/dino/layer_{k}" for k in range(12)}
models = {}
models["DINOv2_last"] = "../../../derive_theoretical_RDMs/more_NNs/dino/last_layer"

# Skeletal representations
models["Ayzenberg Lourenco"] = "../../../derive_theoretical_RDMs/skeletons/ayzenberg_lourenco_2019.csv"
models["Morfoisse Izard"] = "../../../derive_theoretical_RDMs/skeletons/morfoisse_izard_2021.csv"

# Original models
models["CORnet IT"] = "../../../derive_theoretical_RDMs/CNN/output/diss_mat_model-cornet_s_layer-IT.csv"
models["Symbolic"] = "../../../derive_theoretical_RDMs/symbolic/symbolic_sym_diss_mat.csv"

# Additional CNNs
models["densenet"] = "../../../derive_theoretical_RDMs/CNN/output/diss_mat_model-densenet_layer-norm5.csv"
models["resnet"] = "../../../derive_theoretical_RDMs/CNN/output/diss_mat_model-resnet_layer-avgpool.csv"


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

smooth = "smooth"
rdms_movie = rsatoolbox.rdm.rdms.load_rdm(f"./all_rdms/rdms_{cm}_{smooth}.pkl")
times = rdms_movie.subset('subj', ['sub-01']).rdm_descriptors['time']

nmodels = len(models.keys())
all_betas = []
for mname, mpath in models.items():
    mrdm = parse_file(mpath)
    betas = np.zeros((20, 301))
    for isub, sub in enumerate(subs):
        ds = rdms_movie.subset('subj', sub)
        betas[isub, :] = rsatoolbox.rdm.compare(mrdm, scipy.stats.zscore(ds.dissimilarities, axis=1), method=method)[0,:]
    all_betas.append(betas)

all_betas = np.array(all_betas)

plt.close('all')
tmin, tmax = 0, .8
imin, imax = int((tmin + .1)/.004), int((tmax + .1)/.004)
offset = .1/.004
fig, axes = plt.subplots(ncols=1, nrows=nmodels, constrained_layout=True, sharex=True, figsize=(5, 1*nmodels))
for i, mname in enumerate(models.keys()):
    axes[i].axhline(y=0, color='dimgray', linestyle='-', linewidth=0.5)
    axes[i].axvline(x=0, color='dimgray', linestyle='-', linewidth=0.5)
    axes[i].axvline(x=1, color='dimgray', linestyle='-', linewidth=0.5)
    axes[i].axvline(x=0.8, linewidth=0.5, linestyle=(0, (5, 10)), color='k')
    r_ = all_betas[i, :, :]
    mu = np.mean(r_, axis=0)
    se = np.std(r_, axis=0)/np.sqrt(20)
    r_for_clust = r_[:,imin:imax+1]
    _, clusters, p_values, _ = mne.stats.permutation_cluster_1samp_test(
            r_for_clust,
            n_permutations=2**13,
            verbose=False,
            tail=1)
    print([mname, i, p_values[p_values < .05]])
    p05 = np.nan * np.zeros((301))
    for x in np.where(p_values < .05)[0]:
        filt = [int(x) for x in clusters[x][0]+offset]
        p05[filt] = mu[filt]
        print(f"From {np.min((np.array(filt)-offset) * .004)*1000}ms to {np.max((np.array(filt)-offset) * .004)*1000}ms")
    p1 = np.nan * np.zeros((301))
    axes[i].plot(times, mu, color="k", linewidth=0.4)
    axes[i].plot(times, p05, color="k")
    axes[i].fill_between(times, p05, np.zeros((301)), alpha=.5, facecolor="k")
    axes[i].set_title(mname)
    pretty_plot(axes[i])

fig.savefig(f"./figs/lm_{cm}_{method}_{smooth}_multiple.svg")
