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

import nilearn.masking

def process_roi(roi, mask, data):
    ltmap = data.get_fdata().copy()
    ltmap[mask != roi] = np.nan

    indices = ltmap > np.nanmin(ltmap)

    lbetas = np.nanmean(data.get_fdata()[indices])

    df_betas = pd.DataFrame({"value": [lbetas]})
    df_betas["roi_id"] = int(roi)
    return df_betas


def main(task, sub):

    base = "../bids_dataset/derivatives/"

    # Prepare list of imgs and associated evts
    df_betas = []
    for theory in ["IT", "symbolic"]:

        data = nilearn.image.load_img(f"{base}rsa/{sub}/func/{sub}_task-{task}_pred-{theory}_method-crossnobis_eval-corr_cov_corr.nii.gz")

        fname = f"{base}rsa/sub-average/pop-adults_task-geometry_theory-{theory}_bootstraprsa.pkl"
        ref_data = pickle.load(open(fname, "rb"))
        mask_data = ref_data["clusters"]
        ref_tmap = ref_data["tmap"].get_fdata()

        wb_mask = nilearn.masking.compute_brain_mask(data).get_fdata()
        mask_data[wb_mask == False] = 0
        rois = np.unique(mask_data)
        rois = rois[rois > 0]

        list_betas = Parallel(n_jobs=-1)(
            delayed(process_roi)(r, mask_data, data)
            for r in rois
        )
        local_df = pd.concat(list_betas, ignore_index=True)
        local_df["theory"] = theory
        df_betas.append(local_df)

    df_betas = pd.concat(df_betas, ignore_index=True)
    df_betas["task"] = task
    df_betas["subject"] = sub
    df_betas.to_csv(f"{base}rsa/{sub}/func/{sub}_task-{task}_rsa_just-betas.csv", index=False)

if __name__ == "__main__":
    fire.Fire(main)
