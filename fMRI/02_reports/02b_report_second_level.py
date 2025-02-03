import glob
import os
import nilearn
import importlib
import myreporting_second

import numpy as np
import gc

import pickle
import pathlib
import fire

import matplotlib
import matplotlib.pyplot as plt

from nilearn.glm.second_level import SecondLevelModel

import sys
sys.path.insert(0,'..')
import contrasts


def main(kids=False):
    data_path = f"../bids_dataset/derivatives"

    sub_folders = glob.glob(f'{data_path}/nilearn/sub-*/')
    sub_labels = [os.path.basename(s[:-1]).split('-')[1] for s in sub_folders]
    tasks = None
    strkids = "kids"
    if kids:
        sub_labels = sorted(list(set([int(x) for x in sub_labels if int(x) > 300])))
        tasks = ["category", "geometry"]
    else:
        strkids = "adults"
        sub_labels = sorted(list(set([int(x) for x in sub_labels if int(x) < 300])))
        tasks = ["category", "geometry", "geometryHard"]

    for task in tasks:
        models = []
        if task == "geometry":
            sub_labels = [x for x in sub_labels if x != 316]
        for sub_i in sub_labels:
            sub = f"sub-{sub_i}"
            base_fname = (
                f"{data_path}/nilearn/{sub}/func/{sub}_task-{task}"
            )
            model = pickle.load(open(f"{base_fname}_model-spm_full-false.pkl", "rb"))
            models.append(model)
        second_level_model = SecondLevelModel(smoothing_fwhm=8.0, n_jobs=8)
        second_level_model = second_level_model.fit(models)
        cts = {k: nilearn.glm.expression_to_contrast_vector(s, models[0].design_matrices_[0].columns) for k,s in contrasts.contrasts[task].items()}
        n = models[0].design_matrices_[0].shape[1]
        cts["max_var"] = np.eye(n)[n-11:n-6:1]

        report = myreporting_second.make_glm_report(second_level_model, cts, alpha=.001, height_control="fpr")

        report.save_as_html(
            f"./report/{strkids}_second-level_task-{task}_func-report.html"
        )
        del report
        del cts
        del models
        del second_level_model
        gc.collect()

if __name__ == "__main__":
    fire.Fire(main)
