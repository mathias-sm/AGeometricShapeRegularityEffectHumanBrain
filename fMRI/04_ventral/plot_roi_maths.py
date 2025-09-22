import nilearn
import nilearn.plotting
import scipy
import nilearn.image
import numpy as np
import pickle
import fire
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import ListedColormap, BoundaryNorm

task = "category"

base = "../bids_dataset/derivatives"
fmriprep_path = f"{base}/fmriprep/"
nilearn_path = f"{base}/nilearn/"

colors = (["#00B67E"] * 10) + (["#FFAF37"] * 3) + ["#F24000"]
cmap = ListedColormap(colors, name="one_two_three")


def main(age_group="adults"):
    fout = open(f"../bids_dataset/derivatives/bootstrap_clusters/tables/ventral_{age_group}.csv", "w")
    overall_tmap = None
    for idx, mask_name in enumerate(["shape1", "c_number"]):
        # Reference tmap: does _not_ depend on the subject!
        fname = f'{base}/bootstrap_clusters/{age_group}_task-{task}_ctr-{mask_name}.pkl'
        ref_data = pickle.load(open(fname, "rb"))
        if overall_tmap is None:
            overall_tmap = 0. * ref_data["tmap"].get_fdata().copy()

        _, ref_t = nilearn.glm.thresholding.threshold_stats_img(ref_data["tmap"], alpha=.01, height_control="fpr")
        ref_tmap = ref_data["tmap"].get_fdata().copy()

        ref_tmap[ref_tmap < ref_t] = np.nan
        # ref_tmap[:, 58:, :] = np.nan

        overall_tmap += (idx+1) * (ref_tmap > 0)

    to_plot = nilearn.image.new_img_like(ref_data["tmap"], overall_tmap)

    bg = nilearn.datasets.load_mni152_template(1)
    #to_plot = nilearn.image.resample_to_img(to_plot, bg)
    for z in [2, 52, 60]:
        display = nilearn.plotting.plot_stat_map(to_plot,
                title=None,
                bg_img=bg,
                display_mode="z",
                cut_coords=[z],
                annotate=False,
                colorbar=False,
                black_bg=False,
                #threshold=ref_t,
                cmap=cmap,
                symmetric_cbar=False,
                )
        display.savefig(f"{base}/bootstrap_clusters/figures_ventral/{age_group}_task-{task}_z-{z}_merged.png", dpi=100)
        display.close()

if __name__ == "__main__":
    fire.Fire(main)
