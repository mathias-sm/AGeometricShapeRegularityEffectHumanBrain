import mne
from rsatoolbox.data import TemporalDataset
from rsatoolbox.rdm import calc_rdm_movie, combine
from joblib import Parallel, delayed
from scipy.signal import savgol_filter

subs = [f"sub-{(i+1):02}" for i in range(20)]
ordered = ["square", "rectangle", "isoTrapezoid", "parallelogram", "losange", "kite", "rightKite", "rustedHinge", "hinge", "trapezoid", "random"]
cm = "crossnobis"

def estimate_rdm(s, smooth):
    e = mne.read_epochs(f"../../bids_data/derivatives/msm/{s}/meg/{s}_task-POGS_proc-clean+meta_epo.fif")["reference"].as_type("mag").pick_types(meg="mag")
    edata = e.get_data(copy=False)
    data = TemporalDataset(
            edata if not smooth else savgol_filter(edata, 25, 2, axis=2),
            descriptors = {"subj": s},
            obs_descriptors = {"shape": e.metadata["base_shape"], "run": e.metadata["run"]},
            channel_descriptors = {"channels": e.ch_names},
            time_descriptors = {"time": e.times})
    return calc_rdm_movie(data, method=cm, descriptor="shape", cv_descriptor="run")

for smooth in [True, False]:
    rdms_movie = Parallel(n_jobs=2)(delayed(estimate_rdm)(s, smooth) for s in subs)
    rdms_movie = combine.from_partials(rdms_movie, descriptor="shape")
    rdms_movie.sort_by(shape=ordered)
    rdms_movie.save(f"./all_rdms/rdms_{cm}_{'smooth' if smooth else 'unsmooth'}.pkl", file_type="pkl", overwrite=True)
