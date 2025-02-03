import glob
import os
import nilearn
nilearn.EXPAND_PATH_WILDCARDS = True
import importlib
import myreporting

import numpy as np

import pickle
import pathlib

import fire

import sys
sys.path.insert(0,'..')
import contrasts

data_path = f"../bids_dataset/derivatives"

def main(task, subject):
    target_affine = np.diag((2, 2, 2))
    anat = nilearn.image.load_img(
        f"{data_path}/fmriprep/{subject}/anat/{subject}_space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz"
    )
    anat = nilearn.image.resample_img(anat, target_affine)
    base_fname = (
        f"{data_path}/nilearn/{subject}/func/{subject}_task-{task}"
    )
    model = pickle.load(open(f"{base_fname}_model-spm_full-false.pkl", "rb"))
    title = f"Report for subject '{subject}'; task '{task}'"

    cts = {k: nilearn.glm.expression_to_contrast_vector(s, model.design_matrices_[0].columns) for k,s in contrasts.contrasts[task].items()}
    n = model.design_matrices_[0].shape[1]
    cts["max_var"] = np.eye(n)[n-11:n-6:1]

    report = myreporting.make_glm_report(
        model, cts, title=title, bg_img=anat
    )
    report.save_as_html(
        f"./report/{subject}_task-{task}_func-report.html"
    )

if __name__ == "__main__":
    fire.Fire(main)
