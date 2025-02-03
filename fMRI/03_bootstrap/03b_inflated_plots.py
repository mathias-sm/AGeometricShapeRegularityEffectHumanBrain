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

data_path = "../bids_dataset/derivatives"

def main(task="category", contrast="ffa", kids=False):
    iskids = 'kids' if kids else 'adults'

    ref_data = pickle.load(open(f'{data_path}/bootstrap_clusters/{iskids}_task-{task}_ctr-{contrast}.pkl', "rb"))
    ref_tmap = ref_data["tmap"]
    _, ref_t = threshold_stats_img(ref_tmap, alpha=.001, height_control="fpr")
    wb_mask = nilearn.masking.compute_brain_mask(ref_tmap).get_fdata()
    clusters = ref_data["clusters"]
    plot_tmap = np.nan * np.zeros(ref_tmap.dataobj.shape)

    for roi in np.unique(ref_data["clusters"]):
       if roi > 0:
           if ref_data["clust_pvals"][roi - 1] < .05:
               plot_tmap[clusters == roi] = ref_tmap.dataobj[clusters == roi]

    plot_tmap[wb_mask == False] = np.nan
    output_img = new_img_like(ref_tmap, plot_tmap)

    print("All loaded")

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
            surf_fs7=surface.vol_to_surf(output_img, fs7.pial_left)
            fs = fs7.infl_left
        else:
            surf_fs7=surface.vol_to_surf(output_img, fs7.pial_right)
            fs = fs7.infl_right
        display = plotting.plot_surf(
            fs,
            surf_fs7,
            view=v,
            hemi=h,
            title=None,
            threshold=ref_t,
            cmap='cold_hot',
            bg_map=curv_sign_fs7[h] + 2*curv_norm_fs7[h],
            bg_on_data=False,
            scale_bg_map=False,
            symmetric_cmap=True,
            vmax=7,
            vmin=-7,
            # engine="plotly"
        )
        display.savefig(f"{data_path}/bootstrap_clusters/figures/{iskids}_plot-surfInf_second-level_task-{task}_contrast-{contrast}_view-{v}_hemisphere-{h}.png", dpi=400)


    display2 = plotting.plot_stat_map(
           output_img,
           # title=title,
           annotate=False,
           colorbar=False,
           display_mode="z",
           vmax=7,
           cut_coords=range(-20,70,10),
           threshold=ref_t)
    display2.savefig(f"{data_path}/bootstrap_clusters/figures/{iskids}_plot-slices_second-level_task-{task}_contrast-{contrast}.png", dpi=400)

if __name__ == "__main__":
    fire.Fire(main)
