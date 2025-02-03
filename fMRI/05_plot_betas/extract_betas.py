import copy
import pickle
import numpy as np
import pandas as pd

import fire
from joblib import Parallel, delayed

import sys
sys.path.insert(0,'..')
import contrasts

import nilearn
import nilearn.image
from nilearn.interfaces.bids import get_bids_files


def process_roi(roi, mask_name, mask, all_betas, nruns, tmaps, ref_tmap):
    all_mu_betas = []
    n_betas = all_betas[0][0].shape[3]

    # cut reference tmap, used to know if global cluster is >0 or <0
    lref_tmap = ref_tmap.copy()
    lref_tmap[mask != roi] = np.nan
    positive_cluster = (np.nansum(lref_tmap) > 0)

    cst = 0
    for run in range(nruns):
        indices = None
        ltmap = tmaps[run].copy()
        ltmap[mask != roi] = np.nan
        if positive_cluster:
            indices = (ltmap > np.nanpercentile(ltmap, 90))
        else:
            indices = (ltmap < np.nanpercentile(ltmap, 10))

        # Then for every other run, select from this cutoff and average
        for orun in range(nruns):
            if orun != run:
                # For each index, select the betas, then store the average per category
                lbetas = all_betas[orun][0][indices,:]
                all_mu_betas.append(np.mean(lbetas, axis=0))
                cst = cst + 1

    beta_values = np.mean(np.array(all_mu_betas), axis=0)
    df_betas = pd.DataFrame({"value": beta_values, "name": all_betas[run][1]})
    df_betas["roi_contrast"] = mask_name
    df_betas["roi_id"] = int(roi)
    print(f"In total, for some subject, for contrast {mask_name}, roi {roi}, gave back {cst} values. Does that seem right?")
    return df_betas


def main(task, sub, ref_task=None):
    base = "../bids_dataset/derivatives"
    fmriprep_path = f"{base}/fmriprep/"
    nilearn_path = f"{base}/nilearn/"

    if ref_task is None:
        ref_task = task

    # Prepare list of imgs and associated evts
    filters = [("desc", "preproc"), ("space", "MNI152NLin2009cAsym"), ("task", task)]
    pred = get_bids_files( nilearn_path, modality_folder="func", sub_label=sub[4:], filters=[("task", task)])
    ref_img = None
    if sub != "sub-316":
        ref_img = nilearn.image.load_img(f"{fmriprep_path}/{sub}/func/{sub}_task-{task}_run-01_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz")
    else:
        ref_img = nilearn.image.load_img(f"{fmriprep_path}/{sub}/func/{sub}_task-{task}_run-02_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz")
    fmri_glm = pickle.load(open(pred[0], "rb"))

    # Prepare variables to be populated with runs
    all_betas = []
    nruns = len(fmri_glm.results_)
    for run in range(nruns):
        n_vox = fmri_glm.labels_[run].size
        list_betas = fmri_glm.design_matrices_[run].columns
        effect = np.zeros((len(list_betas), n_vox))
        for label_ in fmri_glm.results_[run]:
            label_mask = fmri_glm.labels_[run] == label_
            effect[:, label_mask] = fmri_glm.results_[run][label_].theta
        effect = fmri_glm.masker_.inverse_transform(effect).get_fdata()
        all_betas.append((effect, fmri_glm.design_matrices_[run].columns))

    masks = []
    if task == "category":
        masks = ["vwfa", "ffa", "tool", "house", "shape1", "shape3", "shape1b", "shape3b", "c_number"]
    elif task == "geometry" and ref_task == "category":
        masks = ["shape1"]
    elif task == "geometryHard" and ref_task == "category":
        masks = ["shape1"]
    else:
        masks = ["geom_theory", "geom_behavior_online", "geom_behavior_scanner"]

    df_betas = []
    for mask_name in masks:
        # Reference tmap: does _not_ depend on the subject!
        fname = f'{base}/bootstrap_clusters/adults_task-{ref_task}_ctr-{mask_name}.pkl'
        ref_data = pickle.load(open(fname, "rb"))
        mask_data = ref_data["clusters"]
        ref_tmap = ref_data["tmap"].get_fdata()

        pred = get_bids_files(nilearn_path, modality_folder="func", sub_label=sub[4:], filters=[("task", ref_task)])
        ref_fmri_glm = pickle.load(open(pred[0], "rb"))

        cst = contrasts.contrasts[ref_task][mask_name]
        tmaps=[]
        list_betas = ref_fmri_glm.design_matrices_[0].columns
        con_val = nilearn.glm.contrasts.expression_to_contrast_vector(cst, list_betas)
        for run in range(nruns):
            # Compute the contrast for each run
            lc = nilearn.glm.contrasts.compute_contrast(ref_fmri_glm.labels_[run], ref_fmri_glm.results_[run], con_val)
            lc = ref_fmri_glm.masker_.inverse_transform(lc.z_score())
            tmaps.append(lc.get_fdata())

        # Force load to pickle
        wb_mask = nilearn.masking.compute_brain_mask(ref_img).get_fdata()
        mask_data[wb_mask == False] = 0
        rois = np.unique(mask_data)
        rois = rois[rois > 0]

        list_betas = Parallel(n_jobs=-1)(
            delayed(process_roi)(r, mask_name, mask_data, all_betas, nruns, tmaps, ref_tmap)
            for r in rois
        )
        df_betas.append(pd.concat(list_betas, ignore_index=True))

    df_betas = pd.concat(df_betas, ignore_index=True)
    df_betas["task"] = task
    df_betas["subject"] = sub
    df_betas.to_csv(f"{base}/extracted_betas/{sub}_task-{task}_reftask-{ref_task}_just-betas.csv", index=False)

if __name__ == "__main__":
    fire.Fire(main)
