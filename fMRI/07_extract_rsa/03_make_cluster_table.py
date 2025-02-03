import nilearn.reporting
import nilearn.glm.thresholding
from nilearn.image import new_img_like

import fire

import pickle

import numpy as np

base = "../bids_dataset/derivatives/"

def main(pop, task):

    models = ["symbolic", "IT"]
    paths = {k: f'{base}rsa/sub-average/pop-{pop}_task-{task}_theory-{k}_bootstraprsa.pkl' for k in models}

    for idxt, t in enumerate(models):
        ref_data = pickle.load(open(paths[t], "rb"))

        mask = ref_data["clusters"].copy()

        mask_l = np.array([mask == (i+1) for i in range(len(ref_data["clust_pvals"]))])

        idxs = [i for i,x in enumerate(ref_data["clust_pvals"])]

        try:
            merged = np.sum(1*mask_l[idxs, :, :, :], axis=0) == 1
            masked_tmap = ref_data["tmap"].get_fdata().copy()
            masked_tmap[merged == False] = 0
            masked_tmap = nilearn.image.new_img_like(ref_data["tmap"], masked_tmap)

            _, ref_t = nilearn.glm.thresholding.threshold_stats_img(ref_data["tmap"], alpha=.001, height_control="fpr")

            table = nilearn.reporting.get_clusters_table(masked_tmap, ref_t, two_sided=False)
            table["pval"] = ""
            table["idxs"] = 0

            for i in idxs:
                pv = ref_data["clust_pvals"][i]
                mask = np.array(ref_data["clusters"] != (i+1))
                masked = ref_data["tmap"].get_fdata().copy()
                masked[mask] = np.nan
                max_t = np.nanmax(masked)
                weight = np.nansum(masked)
                if max_t < 0:
                    max_t = np.nanmin(masked)
                x,y,z = np.where(masked == max_t)
                x, y, z = nilearn.image.coord_transform(x, y, z, ref_data["tmap"].affine)
                table.loc[(table["Y"] == y[0]) & (table["X"] == x[0]) & (table["Z"] == z[0]), "pval"] = pv
                table.loc[(table["Y"] == y[0]) & (table["X"] == x[0]) & (table["Z"] == z[0]), "weight"] = weight
                table.loc[(table["Y"] == y[0]) & (table["X"] == x[0]) & (table["Z"] == z[0]), "idxs"] = int(i+1)

            table.to_csv(f'{base}rsa/sub-average/tables/pop-{pop}_task-{task}_theory-{t}_table_full.csv')
        except IndexError:
            print(f"Nothing to find here ;(")



if __name__ == "__main__":
    fire.Fire(main)
