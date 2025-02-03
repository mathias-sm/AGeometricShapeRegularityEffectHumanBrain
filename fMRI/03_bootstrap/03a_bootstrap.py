import glob
import os
import nilearn
import scipy
import copy

import pandas as pd

from joblib import Parallel, delayed, cpu_count

from scipy import ndimage

from nilearn import plotting
from nilearn.glm.thresholding import threshold_stats_img

import numpy as np
import gc
from nilearn.image import new_img_like

import pickle

import argparse

import sys
sys.path.insert(0,'..')

import os.path

import contrasts

from tqdm import trange

import fire
from nilearn.glm.second_level import SecondLevelModel


data_path = "../bids_dataset/derivatives"

def main(task="category", kids=False, n_boot=10000):

    iskids = 'kids' if kids else 'adults'
    sub_folders = glob.glob(f"{data_path}/nilearn/sub-*/")
    sub_labels = [os.path.basename(s[:-1]).split("-")[1] for s in sub_folders]

    if kids:
        sub_labels = sorted(
            list(set([int(x) for x in sub_labels if int(x) > 300]))
        )
        if task == "geometry":
            sub_labels = [x for x in sub_labels if x != 316]
    else:
        sub_labels = sorted(
            list(set([int(x) for x in sub_labels if int(x) < 300]))
        )

    models = []
    columns = None
    for sub_i in sub_labels:
        sub = f"sub-{sub_i}"
        base_fname = f"{data_path}/nilearn/{sub}/func/{sub}_task-{task}"
        model = pickle.load(
            open(f"{base_fname}_model-spm_full-false.pkl", "rb")
        )
        if columns is None:
            columns = model.design_matrices_[0].columns
        models.append(model)

    cts = {
        k: nilearn.glm.expression_to_contrast_vector(s, columns)
        for k, s in contrasts.contrasts[task].items()
    }

    ref_img = None
    wb_mask = None
    for cname, cvec in cts.items():
        if cname in ["any"]:
            continue
        fname = f'./{data_path}/bootstrap_clusters/{iskids}_task-{task}_ctr-{cname}.pkl'
        if os.path.isfile(fname):
            print(f"Skipping contrast {cname}")
        else:
            print(f"Dealing with contrast {cname}!")
            fname = f'./{data_path}/bootstrap_clusters/{iskids}_task-{task}_ctr-{cname}.pkl'

            second_level_model = SecondLevelModel(smoothing_fwhm=8.0, n_jobs=-1)
            second_level_input = [fl.compute_contrast(cvec) for fl in models]

            design_matrix = pd.DataFrame(np.ones((len(models))), columns=["intercept"])
            second_level_model = second_level_model.fit(second_level_input, design_matrix=design_matrix)

            ref_tmap = second_level_model.compute_contrast(second_level_contrast="intercept")
            ref_stat = ref_tmap.get_fdata()

            _, ref_t = threshold_stats_img(ref_tmap, alpha=.001, height_control="fpr")
            ref_out, ref_n_labels = ndimage.label(1*(np.abs(ref_stat) > ref_t))
            ref_cls = [ref_stat[ref_out == i + 1] for i in range(ref_n_labels)]
            ref_cls = [np.sum(x) for x in ref_cls]

            def one_contrast(seed, data):
                # Trick to avoid sharing random numbers across things
                rng = np.random.RandomState(None)
                second_level_model = SecondLevelModel(smoothing_fwhm=8.0, n_jobs=1)
                design_matrix = pd.DataFrame(1-2*rng.randint(0,2,size=len(data)), columns=["intercept"])
                second_level_model = second_level_model.fit(data, design_matrix=design_matrix)
                stat = second_level_model.compute_contrast(second_level_contrast="intercept").get_fdata()

                # The 1* trick is required because labeling may fail on bools
                out, n_labels = ndimage.label(1*(np.abs(stat) > ref_t))
                if n_labels == 0:
                    # If there are _no_ clusters, return 0
                    return 0
                else:
                    # Otherwise return the "mass" of the strongest one
                    cls = [stat[out == i + 1] for i in range(n_labels)]
                    cls = np.array([np.sum(x) for x in cls])
                    return np.max(np.abs(cls))

            results = Parallel(n_jobs=-1)(delayed(one_contrast)(i, second_level_input) for i in trange(n_boot))
            distribution = np.array(results)

            pvals = np.array([np.sum(np.abs(x) < distribution)/n_boot for x in ref_cls])
            clust_and_signif = {}
            clust_and_signif["tmap"] = ref_tmap
            clust_and_signif["clusters"] = ref_out
            clust_and_signif["bootstrap_distribution"] = distribution
            clust_and_signif["clust_pvals"] = pvals
            pickle.dump(clust_and_signif, open(fname, "wb"))

            idxs = [i for i, x in enumerate(ref_cls) if pvals[i] < .05]
            print(f"Done with contrast {cname}! Found the following signif. clusters:")
            print([(i, pvals[i]) for i in idxs])


if __name__ == "__main__":
    fire.Fire(main)
