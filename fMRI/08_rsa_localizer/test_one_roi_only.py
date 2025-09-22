import fire
import pickle
import numpy as np
import nilearn.plotting
from rsatoolbox.data.dataset import Dataset
from rsatoolbox.rdm.calc import calc_rdm
from nilearn.glm.contrasts import expression_to_contrast_vector, compute_contrast

data_path = "../bids_dataset/derivatives"

mapping = {
    "adults": {
        "rIPS": 28,
        "lIPS": 3,
        "rITG": 29,
        "lVent": 4,
        "rVent": 24,
    },
    "kids": {
        "rIPS": 31,
        "lIPS": 1,
        "rITG": 34,
        "lVent": 3,
        "rVent": 22,
    }
}

def main(sub, method="crossnobis", pop="adults"):
    ref_data = pickle.load(open(f'{data_path}/bootstrap_clusters/{pop}_task-category_ctr-shape1.pkl', "rb"))

    glm = pickle.load(open(f"{data_path}/nilearn/{sub}/func/{sub}_task-category_model-spm_full-false.pkl", "rb"))
    conditions = ['shape1', 'shape3', "number", "word", 'Chinese', 'face', 'house', 'tool']

    for area, area_idx in mapping[pop].items():
        mask_area = ref_data["clusters"] == area_idx
        condition_list, runs_list, betas_list = [], [], []
        for condition in conditions:
            ctr = expression_to_contrast_vector(condition, glm.design_matrices_[0].columns)
            for run in range(len(glm.labels_)):
                cst = compute_contrast(glm.labels_[run], glm.results_[run], ctr)
                beta = glm.masker_.inverse_transform(cst.effect_size()).get_fdata()
                betas_list.append(beta[mask_area])
                condition_list.append(condition)
                runs_list.append(run+1) # To start runs from 1

        ds = Dataset(
            np.array(betas_list),
            descriptors={"sub": sub},
            obs_descriptors={"condition": condition_list, "run": runs_list},
        )

        rdm = calc_rdm(ds, method=method, descriptor="condition", cv_descriptor="run")
        rdm.sort_by(condition=conditions)
        pickle.dump(rdm, open(f"{data_path}/rsa/{sub}/func/{sub}_task-category_{area}_{method}_pop-{pop}.pkl", "wb"))

if __name__ == "__main__":
    fire.Fire(main)
