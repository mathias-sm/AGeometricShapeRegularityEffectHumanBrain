import numpy as np
import pandas as pd
from scipy.stats import zscore

import pickle
import fire

from joblib import Parallel, delayed

from rsatoolbox.rdm import RDMs
from rsatoolbox.model import ModelWeighted
from rsatoolbox.model.fitter import fit_regress
import rsatoolbox
from mne import read_source_estimate

from rich.progress import track

color = ["#f1a340", "#998ec3"]
subs = [f"sub-{(i+1):02}" for i in range(20)]

ordered = [
    "square",
    "rectangle",
    "isoTrapezoid",
    "parallelogram",
    "losange",
    "kite",
    "rightKite",
    "rustedHinge",
    "hinge",
    "trapezoid",
    "random",
]

def parse_file(fname):
    """
    Parse an RDM csv file. I reorder things systematically to ensure that
    things end up in the same order across different RDMs.
    """
    diss_content = pd.read_csv(fname, index_col=0)
    diss_content = diss_content.reindex(index=ordered, columns=ordered)
    diss = diss_content.values
    tri_diss = diss[np.triu(np.ones_like(diss, dtype=bool), k=1)]
    rdm = rsatoolbox.rdm.RDMs(
            zscore(tri_diss),
            pattern_descriptors={"shape": ordered})
    rdm.sort_by(shape=ordered)
    return rdm

symbolic = parse_file("../../../derive_theoretical_RDMs/symbolic/symbolic_sym_diss_mat.csv")
dino_last = parse_file("../../../derive_theoretical_RDMs/more_NNs/dino/last_layer")
IT = parse_file("../../../derive_theoretical_RDMs/CNN/output/diss_mat_model-cornet_s_layer-IT.csv")
skel_1 = parse_file("../../../derive_theoretical_RDMs/skeletons/ayzenberg_lourenco_2019.csv")
skel_2 = parse_file("../../../derive_theoretical_RDMs/skeletons/morfoisse_izard_2021.csv")
models = [symbolic, dino_last, IT, skel_1, skel_2]
mnames = ["symbolic", "dino_last", "IT", "skel_1", "skel_2"]

def main(s):
    based = "../../bids_data/derivatives"
    basemne = f"{based}/mne-bids-pipeline/{s}/meg/{s}_task-POGS_"
    basemsm = f"{based}/msm/{s}/meg/{s}_task-POGS_"

    all_rdms = pickle.load(open(f"{basemsm}proc-rsa+crossnobis_rdm_correct_order.pkl", "rb"))
 
    times = np.arange(-0.1, 1.1, 0.004)

    def model_one_rdm(rdm):
        rdm.sort_by(shape=ordered)
        fitted = np.zeros((len(models), len(rdm)))
        for t, lrdm in enumerate(rdm):
            lrdm.dissimilarities[0] = zscore(lrdm.dissimilarities[0])
            for im, m in enumerate(models):
                fitted[im, t] = rsatoolbox.rdm.compare(m, lrdm, method="corr_cov")[0][0]
        return fitted

    fitted = Parallel(n_jobs=2)(delayed(model_one_rdm)(rdm) for rdm in track(all_rdms))
    fitted = np.array(fitted)

    refstc = f"{basemne}shape+eLORETA+hemi"
    refstc = read_source_estimate(refstc)

    for idx_mdl, name in enumerate(mnames):
        this_rtc = refstc.copy()
        this_rtc.data = fitted[:, idx_mdl, :]
        this_rtc.save(f"{basemsm}rsa+eLORETA+many+{name}", overwrite=True)

    print(f"Done with {s}")


if __name__ == "__main__":
    fire.Fire(main)
