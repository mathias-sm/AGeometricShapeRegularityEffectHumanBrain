import fire
import pickle
import numpy as np
from rsatoolbox.data.dataset import Dataset
from rsatoolbox.rdm.calc import calc_rdm
from nilearn.glm.contrasts import expression_to_contrast_vector, compute_contrast
import nilearn
import nilearn.plotting
import nilearn.image

data_path = "../bids_dataset/derivatives"

mapping = {
        "rIPS": 28,
        "lIPS": 3,
        "rITG": 29,
        "lVent": 4,
        "rVent": 24,
}

def main(sub):
    ref_data = pickle.load(open(f'{data_path}/bootstrap_clusters/adults_task-category_ctr-shape1.pkl', "rb"))

    glm = pickle.load(open(f"{data_path}/nilearn/{sub}/func/{sub}_task-category_model-spm_full-false.pkl", "rb"))
    conditions = ['shape1', 'shape3', "number", "word", 'Chinese', 'face', 'house', 'tool']

    for area, area_idx in mapping.items():
        mask_area = 1*(ref_data["clusters"] == area_idx)
        view = nilearn.plotting.view_img(nilearn.image.new_img_like(ref_data["tmap"], mask_area))
        view.save_as_html(f"figs/{area}.html")

if __name__ == "__main__":
    fire.Fire(main)
