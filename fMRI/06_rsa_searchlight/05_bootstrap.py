import glob
import os
import scipy

from nilearn.image import load_img, new_img_like, smooth_img

from nilearn.plotting import view_img, view_img_on_surf
from nilearn.glm.thresholding import threshold_stats_img

import numpy as np
import pandas as pd

from joblib import Parallel, delayed
from scipy import ndimage

import fire
import pickle

from nilearn.glm.second_level import SecondLevelModel

from rich.progress import track

base = "../bids_dataset/derivatives/"


def main(task, theory, pop, estimate="crossnobis", eval="corr_cov", n_boot=10000):

    sub_folders = None
    if pop == "adults":
        sub_folders = glob.glob(f"{base}rsa/sub-2*/")
    else:
        sub_folders = glob.glob(f"{base}rsa/sub-3*/")

    sub_labels = [os.path.basename(s[:-1]).split("-")[1] for s in sub_folders]
    sub_labels = sorted([int(x) for x in sub_labels if x != "308"])

    fc = []
    for sub_i in sub_labels:
        sub = f"sub-{sub_i}"
        img = load_img(f"{base}rsa/{sub}/func/{sub}_task-{task}_pred-{theory}_method-{estimate}_eval-{eval}_corr.nii.gz")
        fc.append(img)

    second_level_model = SecondLevelModel(smoothing_fwhm=8.0, n_jobs=-1)

    design_matrix = pd.DataFrame(np.ones((len(fc))), columns=["intercept"])
    second_level_model = second_level_model.fit(fc, design_matrix=design_matrix)

    ref_tmap = second_level_model.compute_contrast(second_level_contrast="intercept")
    ref_stat = ref_tmap.get_fdata()

    _, ref_t = threshold_stats_img(ref_tmap, alpha=.001, two_sided=False)

    ref_out, ref_n_labels = ndimage.label(1*(ref_stat > ref_t))
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
        out, n_labels = ndimage.label(1*(stat > ref_t))
        if n_labels == 0:
            # If there are _no_ clusters, return 0
            return 0
        else:
            # Otherwise return the "mass" of the strongest one
            cls = [stat[out == i + 1] for i in range(n_labels)]
            cls = np.array([np.sum(x) for x in cls])
            return np.max(cls)

    results = Parallel(n_jobs=-1)(delayed(one_contrast)(i, fc) for i in track(range(n_boot)))
    distribution = np.array(results)

    pvals = np.array([np.sum(x < distribution)/n_boot for x in ref_cls])
    clust_and_signif = {}
    clust_and_signif["tmap"] = ref_tmap
    clust_and_signif["clusters"] = ref_out
    clust_and_signif["bootstrap_distribution"] = distribution
    clust_and_signif["clust_pvals"] = pvals
    fname = f'{base}rsa/sub-average/pop-{pop}_task-{task}_theory-{theory}_method-{estimate}_eval-{eval}_bootstraprsa.pkl'
    pickle.dump(clust_and_signif, open(fname, "wb"))

    idxs = [i for i, x in enumerate(ref_cls) if pvals[i] < .5]
    print(f"Done with contrast {pop}/{task}/{theory}! Found the following signif. clusters:")
    print([(i, pvals[i]) for i in idxs])

if __name__ == "__main__":
    fire.Fire(main)
