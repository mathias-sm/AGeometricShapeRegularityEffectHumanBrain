import mne
import pandas as pd
import numpy as np

from scipy.stats import t

import matplotlib.pyplot as plt

import pickle

from joblib import Parallel, delayed

subs = [f"sub-{(i+1):02}" for i in range(20)]

idx_ref_vs_out = 3

main_delta = []
for sub in [x for x in subs]:
    res = mne.read_evokeds(f"../bids_data/derivatives/mne-bids-pipeline/{sub}/meg/{sub}_task-POGS_ave.fif", verbose=0)[idx_ref_vs_out]
    main_delta.append(res)

fig = mne.grand_average(main_delta).plot_joint(show=False)
fig[0].savefig("figs/fig_refvsout_0.png")
fig[1].savefig("figs/fig_refvsout_1.png")

tmin, tmax = 0, 1.
imin, imax = int((tmin + .1)/.004), int((tmax + .1)/.004)

for chtype in ["grad"]:
    input_cl = np.array([x.get_data(chtype)[:,imin:imax].transpose() for x in main_delta])
    adj, _ = mne.channels.find_ch_adjacency(main_delta[0].info, ch_type=chtype)
    _,cluster,cluster_pv,_ = mne.stats.spatio_temporal_cluster_1samp_test(input_cl, n_jobs=12, adjacency=adj)
    pickle.dump((cluster,cluster_pv), open(f"cl_{chtype}.pkl", "wb"))
