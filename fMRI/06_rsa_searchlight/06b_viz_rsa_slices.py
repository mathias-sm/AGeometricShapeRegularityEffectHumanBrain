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
        print(t)
        theory = pickle.load(open(paths[t], "rb"))
        ref_img = theory["tmap"]
        all_ref_data = np.zeros(ref_img.get_fdata().shape)
        theory_clusters = theory["clusters"]
        theory_pvals = theory["clust_pvals"]
        for idxp, pv in enumerate(theory_pvals):
            if pv < .05:
                all_ref_data[theory_clusters == (idxp+1)] = 1
        all_ref_datas.append(all_ref_data)

    names = models
    names.append("both")
    names.append("empty")

    # Hack to not leave it _fully_ empty, as then the plos shift sizes :(
    empty = 0.*all_ref_datas[0]
    empty[30,30,30] = 1

    all_ref_datas.append(1.*(all_ref_datas[0] + all_ref_datas[1] > 1))
    all_ref_datas.append(empty)

    for idxt, all_ref_data in enumerate(all_ref_datas):

        clust_img = new_img_like(ref_img, all_ref_data)
        colors = ["#ffffff"] + ([possible_colors[idxt]] * 99)
        cmap = LinearSegmentedColormap.from_list("arbitrary", colors, N=100)

        display = plotting.plot_stat_map(
               clust_img,
               bg_img=nilearn.datasets.load_mni152_template(1),
               # title=title,
               annotate=False,
               black_bg=False,
               cmap=cmap,
               colorbar=False,
               display_mode="z",
               vmax=1,
               cut_coords=range(-20,80,10))
        display.savefig(f"{base}rsa/sub-average/figures/pop-{pop}_task-{task}_method-{method}_eval-{eval}_slice_model-{names[idxt]}.png", dpi=100)


if __name__ == "__main__":
    fire.Fire(main)
