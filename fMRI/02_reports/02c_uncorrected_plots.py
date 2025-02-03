import glob
import os
import nilearn
import scipy
import copy
import itertools

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

import fire
from nilearn.glm.second_level import SecondLevelModel


data_path = "../bids_dataset/derivatives"

def main(task="category", kids=False, thresh=.001):

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
        if cname not in ["shape1", "shape1b", "shape3", "shape3b", "geom_behavior_online"]:
            continue
        second_level_model = SecondLevelModel(smoothing_fwhm=8.0, n_jobs=-1)
        second_level_input = [fl.compute_contrast(cvec) for fl in models]

        design_matrix = pd.DataFrame(np.ones((len(models))), columns=["intercept"])
        second_level_model = second_level_model.fit(second_level_input, design_matrix=design_matrix)

        ref_tmap = second_level_model.compute_contrast(second_level_contrast="intercept")
        output_img, ref_t = threshold_stats_img(ref_tmap, alpha=thresh, height_control="fpr")

        fs7 = nilearn.datasets.fetch_surf_fsaverage(mesh="fsaverage7")
        curv_fs7 = {}
        curv_sign_fs7 = {}
        curv_norm_fs7 = {}
        curv_fs7["left"] = nilearn.surface.load_surf_data(fs7.curv_left)
        curv_sign_fs7["left"] = (np.sign(curv_fs7["left"]) + 1)
        curv_norm_fs7["left"] = curv_fs7["left"] - curv_fs7["left"].min()
        curv_norm_fs7["left"] = curv_norm_fs7["left"] / curv_norm_fs7["left"].max()
        curv_fs7["right"] = nilearn.surface.load_surf_data(fs7.curv_right)
        curv_sign_fs7["right"] = (np.sign(curv_fs7["right"]) + 1)
        curv_norm_fs7["right"] = curv_fs7["right"] - curv_fs7["right"].min()
        curv_norm_fs7["right"] = curv_norm_fs7["right"] / curv_norm_fs7["right"].max()
        for v, h in itertools.product(['lateral', 'ventral', 'medial'], ['left', 'right']):
            surf_fs7 = None
            fs = None
            if h=="left":
                surf_fs7=nilearn.surface.vol_to_surf(output_img, fs7.pial_left)
                fs = fs7.infl_left
            else:
                surf_fs7=nilearn.surface.vol_to_surf(output_img, fs7.pial_right)
                fs = fs7.infl_right
            display = plotting.plot_surf(
                fs,
                surf_fs7,
                view=v,
                hemi=h,
                title=None,
                threshold=ref_t,
                cmap='cold_hot',
                bg_map=curv_sign_fs7[h] + 2*curv_norm_fs7[h],
                bg_on_data=False,
                scale_bg_map=False,
                symmetric_cmap=True,
                vmax=7,
                vmin=-7,
                # engine="plotly"
            )
            display.savefig(f"./figs_uncorr/{iskids}_plot-surfInf_second-level_thresh-{thresh}_task-{task}_contrast-{cname}_view-{v}_hemisphere-{h}.png", dpi=400)




if __name__ == "__main__":
    fire.Fire(main)
