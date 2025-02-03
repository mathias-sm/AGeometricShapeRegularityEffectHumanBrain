import numpy as np
from rsatoolbox.util.searchlight import get_volume_searchlight
from rsatoolbox.data.dataset import Dataset
from nilearn.image import load_img
from nilearn.masking import compute_brain_mask
from rsatoolbox.rdm.calc import calc_rdm
from rsatoolbox.rdm import RDMs
import pickle
from rich.progress import track
from os import path, makedirs
import fire
from nilearn.glm.contrasts import expression_to_contrast_vector, compute_contrast
from joblib import Parallel, delayed

shapes_easy = ["square", "rectangle", "isoTrapezoid", "losange", "hinge", "random"]
shapes_hard = ["square", "rectangle", "isoTrapezoid", "parallelogram", "losange", "kite", "hinge", "random"]

base = "../bids_dataset/derivatives/"

def main(task, sub, method="correlation"):
    # Prepare beta maps
    glm = pickle.load(open(f"{base}nilearn/{sub}/func/{sub}_task-{task}_model-spm_full-false.pkl", "rb"))
    ref_img = load_img(f"{base}/fmriprep/{sub}/func/{sub}_task-{task}_run-01_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz")

    # Prepare searchlight positions
    mask = compute_brain_mask(ref_img).get_fdata()
    centers, neighbors = get_volume_searchlight(mask, radius=3, threshold=0.5)

    # Put the right betas, runs and so on in lists to pass as descriptors.
    # Change "shapes" to instead put the predictors across which you want to
    # estimate dissimilarity. Note that a priori I'm doing cross-validad
    # mahalnobis distance across runs.
    shape_list, runs_list, betas_list = [], [], []
    shapes = shapes_easy
    if task == "geometryHard":
        shapes = shapes_hard

    for shape in shapes:
        ctr = expression_to_contrast_vector(shape, glm.design_matrices_[0].columns)
        for run in range(len(glm.labels_)):
            cst = compute_contrast(glm.labels_[run], glm.results_[run], ctr)
            beta = glm.masker_.inverse_transform(cst.effect_size()).get_fdata()
            beta[mask == 0] = np.nan
            betas_list.append(beta)
            shape_list.append(shape)
            runs_list.append(run+1) # To start runs from 1


    # minimal data plumbing to fit the function: 4d (betas, x, y, z) -> 2d (betas, longlist)
    # Note that the first dimension should be equal to n_predictors * n_run
    data_4d = np.array(betas_list)
    data_2d = np.nan_to_num(data_4d.reshape([data_4d.shape[0], -1]))
    obs_descriptors = {"shape": shape_list, "run": runs_list}

    # From here onward: basically copying
    # https://rsatoolbox.readthedocs.io/en/latest/_modules/rsatoolbox/util/searchlight.html#get_searchlight_RDMs
    # but making it parallel for faster computation, and also because
    # crossnobis requires an extra cv_descriptor argument which is not present
    # with the default function. I might suggest a pull request here.
    # If that doesn't matter (e.g. using correlation distance) one could just replace the next ~20 lines with :
    # SL_RDM = rsatoolbox.util.searchlight.get_searchlight_RDMs(data_2d, centers, neighbors, events, method='crossnobis')
    n_centers = centers.shape[0]
    chunked_center = np.split(np.arange(n_centers), np.linspace(0, n_centers, 101, dtype=int)[1:-1])

    def map_one(chunks):
        center_data = []
        for c in chunks:
            # grab this center and neighbors
            center = centers[c]
            center_neighbors = neighbors[c]
            ds = Dataset(
                data_2d[:, center_neighbors],
                descriptors={"center": center},
                obs_descriptors=obs_descriptors,
                channel_descriptors={"voxels": center_neighbors},
            )
            center_data.append(ds)

        rdm = calc_rdm(center_data, method=method, descriptor="shape")  # unused for correlation: cv_descriptor="run"
        return rdm

    # MAP -> send parallel computation over 10 jobs
    RDM_corrs = Parallel(n_jobs=10)(delayed(map_one)(c) for c in track(chunked_center))

    # REDUCE -> take the result of the parallel computations and put them back together
    n_conds = len(np.unique(obs_descriptors["shape"]))
    RDM = np.zeros((n_centers, n_conds * (n_conds - 1) // 2))
    for chunks, RDM_corr in zip(chunked_center, RDM_corrs):
        RDM[chunks, :] = RDM_corr.dissimilarities

    rdm_des={'voxel_index': centers}
    pat_des=RDM_corrs[0].pattern_descriptors
    SL_RDM = RDMs(RDM, rdm_descriptors=rdm_des, dissimilarity_measure=method, pattern_descriptors=pat_des)
    SL_RDM.sort_by(shape=shapes)

    pickle.dump(SL_RDM, open(f"{base}rsa/{sub}/func/{sub}_task-{task}_method-{method}_RDM.pkl", "wb"))


if __name__ == "__main__":
    fire.Fire(main)
