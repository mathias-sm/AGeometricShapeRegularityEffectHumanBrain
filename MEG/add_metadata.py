import mne
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mne_bids import BIDSPath

shapes = ['rectangle', 'square', 'isoTrapezoid', 'parallelogram', 'losange', 'kite', 'rightKite', 'rustedHinge', 'hinge', 'trapezoid', 'random']
theory_df = pd.read_csv("./symbolic_model_simple_ref.csv")

cplx_theory = {}
for s in shapes:
    cplx_theory[s] = {}
    for out in range(5):
        v = theory_df.loc[(theory_df["name"]==s) & (theory_df["out_type"]==out)]["distances"].values[0]
        cplx_theory[s][out] = v

for sub in [f"sub-{(i+1):02}" for i in range(20)]:
    e = mne.read_epochs(f"./bids_data/derivatives/mne-bids-pipeline/{sub}/meg/{sub}_task-POGS_proc-clean_epo.fif")

    root = f"./bids_data/{sub}/meg/{sub}_task-POGS"
    dfs = []
    for r in range(1, 9):
        run = f"run-{r:02}"
        try:
            df = pd.read_csv(f"{root}_{run}_events.tsv", sep="\t")
            df["run"] = r
            dfs.append(df)
        except FileNotFoundError:
            print(f"Subject {sub} has no run {run}")

    d = pd.concat(dfs)
    d["base_shape"] = [x.split("/")[1] for x in d["trial_type"]]
    local_acc = 0
    last_s = d["base_shape"].iloc[0]
    acc_l = [0]
    for this_s in d["base_shape"].iloc[1:].values:
        if last_s == this_s:
            local_acc = local_acc + 1
            if local_acc > 29: # We went over a run here!
                local_acc = 0
        else:
            local_acc = 0
        last_s = this_s
        acc_l.append(local_acc)

    d["is_outlier"] = [x.split("/")[2] != "reference" for x in d["trial_type"]]
    d["outlier_type"] = [int(x.split("/")[3]) if len(x.split("/"))>3 else 0 for x in d["trial_type"]]
    d["number_props"] = [int(cplx_theory[x][d["outlier_type"].values[i]]) for i, x in enumerate(d["base_shape"])]
    d["delta_prop"] = [int(cplx_theory[x][0] - cplx_theory[x][d["outlier_type"].values[i]]) for i, x in enumerate(d["base_shape"])]
    d["number_props_ref"] = [int(cplx_theory[x][0]) for i, x in enumerate(d["base_shape"])]
    d["trial_number_from_run"] = d.index
    d["trial_number_from_miniblock"] = acc_l
    d["is_early"] = ["early" if x else "late" for x in (d["trial_number_from_miniblock"] < 6)]
    e.metadata = d.iloc[e.selection].copy()
    e.metadata["trial_number_from_start"] = e.selection
    e.metadata["intercept"] = 1
    e.metadata.drop(["duration", "value", "sample", "onset"], axis=1, inplace=True)
    e.save(f"./bids_data/derivatives/msm/{sub}/meg/{sub}_task-POGS_proc-clean+meta_epo.fif", overwrite=True)
