import pickle
from joblib import Parallel, delayed
from mne import read_source_estimate, compute_source_morph

models = ["symbolic", "dino_last", "IT", "skel_1", "skel_2"]
subs = [f"sub-{(i+1):02}" for i in range(20) if i != 2]
based = "../../bids_data/derivatives"

def process_one_sub(s, name):
    basemsm = f"{based}/msm/{s}/meg/{s}_task-POGS_"
    morphed = []
    stc = read_source_estimate(f"{basemsm}rsa+eLORETA+many+{name}")
    morph = compute_source_morph(
            stc,
            subject_from=s,
            subject_to='fsaverage',
            subjects_dir=f"{based}/freesurfer/subjects/")
    return(morph.apply(stc))

morphed = {}
for name in models:
    morphed[name] = Parallel(n_jobs=19)(delayed(process_one_sub)(s, name) for s in subs)

pickle.dump(morphed, open(f"{based}/msm/sub-average/meg/sub-average_task-POGS_all+rsa+many.pkl", "wb"))
