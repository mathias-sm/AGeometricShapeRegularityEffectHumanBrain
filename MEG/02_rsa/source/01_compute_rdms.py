import numpy as np
from rsatoolbox.data import TemporalDataset
from rsatoolbox.rdm import calc_rdm_movie

import pickle

from joblib import Parallel, delayed

from mne import read_epochs, setup_source_space, spatial_dist_adjacency, add_source_space_distances
from mne.minimum_norm import (
    read_inverse_operator,
    apply_inverse,
    prepare_inverse_operator,
)

import fire

from rich.progress import track

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
based = "../../bids_data/derivatives"

def main(s):
    basemsm = f"{based}/msm/{s}/meg/{s}_task-POGS_"
    basemne = f"{based}/mne-bids-pipeline/{s}/meg/{s}_task-POGS_"
    e = read_epochs(f"{basemsm}proc-clean+meta_epo.fif")["reference"]
    m = e.metadata

    # load inverse and compute adjacency (arbitrary decision point here!)
    inv = read_inverse_operator(f"{basemne}inv.fif")
    inv = prepare_inverse_operator(inv, 26, lambda2=1 / 9, method="eLORETA")
    add_source_space_distances(inv["src"])
    adj = spatial_dist_adjacency(inv["src"], dist=0.02)


    # Make this 3d to prepare for rsatoolbox requirements
    # n_obs, n_captors, n_samples where n_obs collapses runs and shapes
    nruns = np.max(m["run"].values)
    data_3d = np.zeros((nruns * 11, inv["nsource"], len(e.times)))
    shapes, runs = [], []
    k = 0
    for run in range(1, 1 + nruns):
        for shape_idx, shape in enumerate(ordered):
            evoked = e[m["run"] == run][shape].average()
            stc = apply_inverse(evoked, inv, method="eLORETA", prepared=True)
            data_3d[k, :, :] = stc.data
            k = k + 1
            shapes.append(shape)
            runs.append(run)

    all_idxs = []
    for center in range(inv["nsource"]):
        # Manually adds the center, ommited otherwise. We can recover it below!
        all_idxs.append(np.append(adj.getcol(center).nonzero()[0], center))

    def calc_one_rdm(idxs):
        # Recover the center
        center = idxs[-1]
        data = TemporalDataset(
            data_3d[:, idxs, :],
            descriptors={"subj": s, "center": center},
            obs_descriptors={"shape": shapes, "run": runs},
            channel_descriptors={"channels": idxs},
            time_descriptors={"time": e.times},
        )
        rdm = calc_rdm_movie(
            data, method="crossnobis", descriptor="shape", cv_descriptor="run"
        )
        rdm.sort_by(shape=ordered)
        return rdm

    # all_rdms = Parallel(n_jobs=3)(delayed(calc_one_rdm)(c) for c in all_idxs)
    all_rdms = [calc_one_rdm(c) for c in track(all_idxs)]
    print(f"{basemsm}proc-rsa+crossnobis_rdm_correct_order.pkl")
    pickle.dump(all_rdms, open(f"{basemsm}proc-rsa+crossnobis_rdm_correct_order.pkl", "wb"))


if __name__ == "__main__":
    fire.Fire(main)
