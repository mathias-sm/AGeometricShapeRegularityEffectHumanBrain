import nilearn
import nilearn.regions
import nilearn.reporting
import itertools
import plotly

from scipy import ndimage

import fire

import sys
sys.path.insert(0,'..')

import pickle

from nilearn import plotting
from nilearn import surface
from nilearn.glm.thresholding import threshold_stats_img

import numpy as np
from nilearn.image import new_img_like

import nilearn.datasets
import contrasts
from matplotlib.colors import LinearSegmentedColormap

base = "../bids_dataset/derivatives/"

theories = {
        "symbolic": "./derive_theoretical_RDMs/symbolic/symbolic_sym_diss_mat.csv",
        "IT": "./derive_theoretical_RDMs/CNN/output/diss_mat_model-cornet_s_layer-IT.csv",
        "dino_last": "./derive_theoretical_RDMs/more_NNs/dino/last_layer",
        "skel_1": "./derive_theoretical_RDMs/skeletons/ayzenberg_lourenco_2019.csv",
        "skel_2": "./derive_theoretical_RDMs/skeletons/morfoisse_izard_2021.csv"
}
theories_names = theories.keys()  # Used to get consistent ordering throughout

def main(pop, task, method="crossnobis", eval="corr_cov"):

    paths = {k: f'{base}rsa/sub-average/pop-{pop}_task-{task}_theory-{k}_method-{method}_eval-{eval}_bootstraprsa.pkl' for k in theories_names}

    all_ref_datas = []
    for idxt, t in enumerate(theories_names):
        theory = pickle.load(open(paths[t], "rb"))
        ref_img = theory["tmap"]
        all_ref_data = np.zeros(ref_img.get_fdata().shape)
        theory_clusters = theory["clusters"]
        theory_pvals = theory["clust_pvals"]
        for idxp, pv in enumerate(theory_pvals):
            if pv < .05:
                all_ref_data[theory_clusters == (idxp+1)] = ref_img.get_fdata()[theory_clusters == (idxp+1)]
        all_ref_datas.append(all_ref_data)

    for idxt, theory_name in enumerate(theories_names):

        all_ref_data = all_ref_datas[idxt]
        clust_img = new_img_like(ref_img, all_ref_data)

        fs7 = nilearn.datasets.fetch_surf_fsaverage(mesh="fsaverage7")
        curv_fs7 = {}
        curv_sign_fs7 = {}
        curv_norm_fs7 = {}
        curv_fs7["left"] = surface.load_surf_data(fs7.curv_left)
        curv_sign_fs7["left"] = (np.sign(curv_fs7["left"]) + 1)
        curv_norm_fs7["left"] = curv_fs7["left"] - curv_fs7["left"].min()
        curv_norm_fs7["left"] = curv_norm_fs7["left"] / curv_norm_fs7["left"].max()
        curv_fs7["right"] = surface.load_surf_data(fs7.curv_right)
        curv_sign_fs7["right"] = (np.sign(curv_fs7["right"]) + 1)
        curv_norm_fs7["right"] = curv_fs7["right"] - curv_fs7["right"].min()
        curv_norm_fs7["right"] = curv_norm_fs7["right"] / curv_norm_fs7["right"].max()

        for v, h in itertools.product(['lateral', 'ventral', 'medial'], ['left', 'right']):
            surf_fs7 = None
            fs = None
            if h=="left":
                surf_fs7=surface.vol_to_surf(clust_img, fs7.pial_left)
                fs = fs7.infl_left
            else:
                surf_fs7=surface.vol_to_surf(clust_img, fs7.pial_right)
                fs = fs7.infl_right
            display = plotting.plot_surf(
                fs,
                surf_fs7,
                view=v,
                hemi=h,
                title=None,
                threshold=.01,
                bg_map=curv_sign_fs7[h] + 2*curv_norm_fs7[h],
                bg_on_data=False,
                scale_bg_map=False,
                cmap="black_red",
                vol_to_surf_kwargs={"interpolation": "nearest"},
                # engine="plotly"
            )
            display.savefig(f"{base}rsa/sub-average/figures/pop-{pop}_task-{task}_method-{method}_eval-{eval}_view-{v}_hemisphere-{h}_model-{theory_name}_isolated.png", dpi=400)
            print(f"{base}rsa/sub-average/figures/pop-{pop}_task-{task}_method-{method}_eval-{eval}_view-{v}_hemisphere-{h}_model-{theory_name}_isolated.png")

if __name__ == "__main__":
    fire.Fire(main)
