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

def process_roi(roi, ref_cst, mask, all_betas, nruns, tmaps, ref_tmap):
    all_mu_betas = []

    # cut reference tmap, used to know if global cluster is >0 or <0
    lref_tmap = ref_tmap.copy()
    lref_tmap[mask != roi] = np.nan
    positive_cluster = (np.nansum(lref_tmap) > 0)

    for run in range(nruns):
        indices = None
        ltmap = tmaps[run].copy()
        ltmap[mask != roi] = np.nan
        if positive_cluster:
            indices = (ltmap > np.nanpercentile(ltmap, 90))
        else:
            indices = (ltmap < np.nanpercentile(ltmap, 10))

        lbetas = np.nanmean(all_betas.get_fdata()[indices])
        all_mu_betas.append(lbetas)

    beta_values = np.nanmean(np.array(all_mu_betas), axis=0)
    df_betas = pd.DataFrame({"value": [beta_values]})
    df_betas["roi_contrast"] = ref_cst
    df_betas["roi_id"] = int(roi)
    return df_betas


def main(task, sub, ref_task, ref_cst):

    base = "../bids_dataset/derivatives/"

    # Prepare list of imgs and associated evts
    df_betas = []
    for theory in ["IT", "symbolic"]:
        data = nilearn.image.load_img(f"{base}rsa/{sub}/func/{sub}_task-{task}_pred-{theory}_corr.nii.gz")

        fname = f'{base}bootstrap_clusters/adults_task-{ref_task}_ctr-{ref_cst}.pkl'
        ref_data = pickle.load(open(fname, "rb"))
        mask_data = ref_data["clusters"]
        ref_tmap = ref_data["tmap"].get_fdata()

        pred = get_bids_files(f"{base}/nilearn/", modality_folder="func", sub_label=sub[4:], filters=[("task", ref_task)])
        ref_fmri_glm = pickle.load(open(pred[0], "rb"))
        nruns = len(ref_fmri_glm.results_)


        cst = contrasts.contrasts[ref_task][ref_cst]
        tmaps=[]
        list_betas = ref_fmri_glm.design_matrices_[0].columns
        con_val = nilearn.glm.contrasts.expression_to_contrast_vector(cst, list_betas)
        for run in range(nruns):
            # Compute the contrast for each run
            lc = nilearn.glm.contrasts.compute_contrast(ref_fmri_glm.labels_[run], ref_fmri_glm.results_[run], con_val)
            lc = ref_fmri_glm.masker_.inverse_transform(lc.z_score())
            tmaps.append(lc.get_fdata())

        # Force load to pickle
        print("about to mask")
        wb_mask = nilearn.masking.compute_brain_mask(data).get_fdata()
        print("masked!")
        mask_data[wb_mask == False] = 0
        rois = np.unique(mask_data)
        rois = rois[rois > 0]

        list_betas = Parallel(n_jobs=1)(
            delayed(process_roi)(r, ref_cst, mask_data, data, nruns, tmaps, ref_tmap)
            for r in rois
        )
        local_df = pd.concat(list_betas, ignore_index=True)
        local_df["theory"] = theory
        df_betas.append(local_df)

    df_betas = pd.concat(df_betas, ignore_index=True)
    df_betas["task"] = task
    df_betas["subject"] = sub
    df_betas["ref_task"] = ref_task
    df_betas.to_csv(f"{base}rsa/{sub}/func/{sub}_task-{task}_reftask-{ref_task}_just-betas.csv", index=False)

if __name__ == "__main__":
    fire.Fire(main)
