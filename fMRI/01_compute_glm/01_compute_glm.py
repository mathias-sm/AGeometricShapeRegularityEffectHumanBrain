import nilearn
from nilearn.glm.first_level import first_level_from_bids, make_first_level_design_matrix
from nilearn.image import get_data, high_variance_confounds, load_img
from nilearn.glm.regression import SimpleRegressionResults
nilearn.EXPAND_PATH_WILDCARDS = True

import os.path

import numpy as np
import pandas as pd

import pickle
import pathlib

import fire

def main(task, subject):
    basepath = "../bids_dataset/derivatives/"
    folder = basepath + "nilearn/" + subject + "/" + "func"
    base_fname = f"{folder}/{subject}_task-{task}"
    model_dump_path = f"{base_fname}_model-spm"
    save_path = f"{model_dump_path}_full-false.pkl"

    pathlib.Path(folder).mkdir(parents=True, exist_ok=True)

    ref_func = None
    mask = None

    mask = load_img(f"{basepath}fmriprep/{subject}/anat/{subject}_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz")
    try:
        ref_func = load_img(f"{basepath}fmriprep/{subject}/func/{subject}_task-{task}_run-01_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz")
    except ValueError:  # sub 316 is missing that one so we have to use the 2nd one :-(
        ref_func = load_img(f"{basepath}fmriprep/{subject}/func/{subject}_task-{task}_run-02_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz")
    mask = nilearn.image.resample_to_img(mask, ref_func, interpolation="nearest")

    (
        models,
        models_run_imgs,
        models_events,
        models_confounds,
    ) = first_level_from_bids(
        "../bids_dataset/",
        task_label=task,
        space_label="MNI152NLin2009cAsym",
        derivatives_folder="derivatives/fmriprep",
        sub_labels=[subject[4:]],
        hrf_model="spm",
        smoothing_fwhm=4.0,
        slice_time_ref=0.5,
        mask_img=mask,
        noise_model="ar2",
        standardize="psc", # The doc says only True/False, but nilearn actually accepts this as "percent of signal change"
        signal_scaling=False,
        drift_model="polynomial",
        drift_order=5,
        t_r=1.81,
        verbose=2,
        n_jobs=-1,
        minimize_memory=True,
    )

    for i in range(len(models)):
        model, imgs, events, confounds = (
            models[i],
            models_run_imgs[i],
            models_events[i],
            models_confounds[i],
        )

        design_matrices = []
        for run in range(len(confounds)):
            print(f"\t- Preparing matrix for run {run}")
            mov_confounds = confounds[run][
                ["trans_x", "trans_y", "trans_z",
                 "rot_x", "rot_y", "rot_z",
                 "csf", "white_matter"]
            ]

            events[run] = events[run][events[run]["trial_type"] != "anypress"]     # Remove the rt predictor
            events[run] = events[run][events[run]["trial_type"] != "sq_anypress"]  # Remove the "squared" rt predictor

            hv_confounds = pd.DataFrame(
                    high_variance_confounds(imgs[run], percentile=1),
                    columns=[f"HighVariance_{x}" for x in range(5)])
            all_confounds = pd.concat([mov_confounds, hv_confounds], axis=1)

            n_scans = get_data(imgs[run]).shape[3]
            start_time = model.slice_time_ref * model.t_r
            end_time = (n_scans - 1 + model.slice_time_ref) * model.t_r
            frame_times = np.linspace(start_time, end_time, n_scans)

            design_matrices.append(make_first_level_design_matrix(
                frame_times,
                events[run],
                model.hrf_model,
                model.drift_model,
                model.high_pass,
                model.drift_order,
                model.fir_delays,
                all_confounds.values,
                all_confounds.columns.tolist(),
                model.min_onset,
            ))

        model.fit(imgs, design_matrices=design_matrices)
        pickle.dump(model, open(save_path, "wb"))


if __name__ == "__main__":
    fire.Fire(main)
