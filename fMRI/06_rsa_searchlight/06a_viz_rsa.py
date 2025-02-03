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

possible_colors = ["#D1790A", "#5A2BA1", "#166629", "#000000"]

def main(pop, task, method="crossnobis", eval="corr_cov"):

    models = ["symbolic", "IT"]
    paths = {k: f'{base}rsa/sub-average/pop-{pop}_task-{task}_theory-{k}_method-{method}_eval-{eval}_bootstraprsa.pkl' for k in models}

    all_ref_datas = []
    for idxt, t in enumerate(models):
        theory = pickle.load(open(paths[t], "rb"))
        ref_img = theory["tmap"]
        all_ref_data = np.zeros(ref_img.get_fdata().shape)
        theory_clusters = theory["clusters"]
        theory_pvals = theory["clust_pvals"]
        for idxp, pv in enumerate(theory_pvals):
            if pv < .1:
                all_ref_data[theory_clusters == (idxp+1)] = 1
        all_ref_datas.append(all_ref_data)

    names = models
    names.append("both")
    names.append("empty")

    all_ref_datas.append(1.*(all_ref_datas[0] + all_ref_datas[1] > 1))
    all_ref_datas.append(0.*(all_ref_datas[0]))

    np.sum(all_ref_datas[2])

    for idxt, all_ref_data in enumerate(all_ref_datas):

        clust_img = new_img_like(ref_img, all_ref_data)
        colors = ["#ffffff"] + ([possible_colors[idxt]] * 99)
        cmap = LinearSegmentedColormap.from_list("arbitrary", colors, N=100)

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
                cmap=cmap,
                bg_map=curv_sign_fs7[h] + 2*curv_norm_fs7[h],
                bg_on_data=False,
                scale_bg_map=False,
                symmetric_cmap=True,
                vmax=1,
                vol_to_surf_kwargs={"interpolation": "nearest"},
                # engine="plotly"
            )
            display.savefig(f"{base}rsa/sub-average/figures/pop-{pop}_task-{task}_method-{method}_eval-{eval}_view-{v}_hemisphere-{h}_model-{names[idxt]}.png", dpi=400)

if __name__ == "__main__":
    fire.Fire(main)
