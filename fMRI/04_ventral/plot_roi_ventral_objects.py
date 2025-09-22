import nilearn
import nilearn.plotting
import scipy
import nilearn.image
import numpy as np
import pickle
import fire
import matplotlib.colors as colors
import matplotlib.pyplot as plt

task = "category"

base = "../bids_dataset/derivatives"
fmriprep_path = f"{base}/fmriprep/"
nilearn_path = f"{base}/nilearn/"

cmap=["PRGn", "PiYG", "RdGy", "RdBu_r"]

def truncate_colormap(cmap, minval=0.0, maxval=0.3, n=100):
   cmap_inv = np.flip(cmap(np.linspace(minval, maxval, n)))
   cmap_str = cmap(np.linspace(minval, maxval, n))
   new_cmap = colors.LinearSegmentedColormap.from_list(
           'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
           np.concatenate([cmap_inv, cmap_inv, cmap_str]))
   return new_cmap

cmap=[truncate_colormap(plt.get_cmap(x)) for x in cmap]

mni_roi_center = {
        "vwfa": (-45, -57, -14),
        "ffa": (40, -55, -14),
}

def main(age_group="adults"):

    fout = open(f"../bids_dataset/derivatives/bootstrap_clusters/tables/ventral_{age_group}.csv", "w")
    for idx, mask_name in enumerate(["ffa", "vwfa", "tool", "house"]):
        # Reference tmap: does _not_ depend on the subject!
        fname = f'{base}/bootstrap_clusters/{age_group}_task-{task}_ctr-{mask_name}.pkl'
        ref_data = pickle.load(open(fname, "rb"))

        mask_data = ref_data["clusters"]

        if mask_name in mni_roi_center.keys():
            x, y, z = mni_roi_center[mask_name]
            x, y, z = nilearn.image.coord_transform(x, y, z, scipy.linalg.inv(ref_data["tmap"].affine))
            xmin = int(np.floor(x)) - 2
            xmax = int(np.ceil(x)) + 2
            ymin = int(np.floor(y)) - 2
            ymax = int(np.ceil(y)) + 2
            zmin = int(np.floor(z)) - 2
            zmax = int(np.ceil(z)) + 2
        else:
            xmin = 0
            xmax = 97
            ymin = 0
            ymax = 49
            zmin = 30
            zmax = 32

        rois = np.unique(mask_data)
        rois = rois[rois > 0]

        keep = []
        for roi in rois:
            one_only = mask_data.copy()
            loc_tmap = ref_data["tmap"].get_fdata().copy()
            loc_tmap[one_only != roi] = np.nan
            one_only[one_only != roi] = 0
            slice = one_only[xmin:xmax,ymin:ymax,zmin:zmax]
            if np.sum(slice) > 0:
                if np.nansum(loc_tmap[xmin:xmax,ymin:ymax,zmin:zmax]) > 0:
                    keep.append(roi)
                    print(",".join([age_group, mask_name, str(roi), f"p={ref_data["clust_pvals"][roi-1]}"]), file=fout)

        ref_tmap = ref_data["tmap"].get_fdata().copy()

        plot_tmap = np.nan * np.zeros((ref_tmap.shape))

        for roi in keep:
            plot_tmap[mask_data == roi] = ref_tmap[mask_data == roi]

        if (age_group == "kids" and mask_name == "vwfa"):
            print("updating!")
            plot_tmap[45,45,:] = .001

        to_plot = nilearn.image.new_img_like(ref_data["tmap"], plot_tmap)
        bg = nilearn.datasets.load_mni152_template(1)
        #to_plot = nilearn.image.resample_to_img(to_plot, bg)
        display = nilearn.plotting.plot_stat_map(to_plot,
                title=None,
                bg_img=bg,
                display_mode="z",
                cut_coords=[-16],
                annotate=False,
                colorbar=False,
                black_bg=False,
                #threshold=ref_t,
                cmap=cmap[idx],
                symmetric_cbar=True,
                )
        display.savefig(f"{base}/bootstrap_clusters/figures_ventral/{age_group}_task-{task}_ctr-{mask_name}_ventral.png", dpi=100)
        display.close()

if __name__ == "__main__":
    fire.Fire(main)
