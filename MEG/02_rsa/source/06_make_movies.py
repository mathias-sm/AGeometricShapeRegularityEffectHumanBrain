import numpy as np
import pickle
import fire
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors
import scipy
import matplotlib.pyplot as plt
import imageio

based = "../../bids_data/derivatives"
basemsm = f"{based}/msm/sub-average/meg/sub-average_task-POGS_"

def make_movie(stc, cmap):
    brain = stc.plot(
            hemi="split",
            colormap = cmap,
            views=["lateral", "medial"],
            subjects_dir=f"{based}/freesurfer/subjects/",
            size=(1920, 1080),
            smoothing_steps=20,
            clim=dict(kind="value", lims=[0,.5,1]),
            transparent=False,
            surface="white",
            show_traces=False,
            time_viewer=False,
            # colorbar=False,
            brain_kwargs=dict(show=False))
    result = brain._make_movie_frames(50, -.01, .6, 20, "cubic", None, False)
    result = np.array(result[10:])
    brain.close()
    return result


morphed = pickle.load(open(f"{based}/msm/sub-average/meg/sub-average_task-POGS_all+rsa+many.pkl", "rb"))

threshold = 0.01
data_IT = np.array([x.data for x in morphed["IT"]])
tt_IT = scipy.stats.ttest_1samp(data_IT, popmean=0, alternative="greater")
ref_stc_data_IT = np.mean(data_IT, axis=0)
ref_stc_data_IT[tt_IT.pvalue > threshold] = 0
ref_stc_data_IT[tt_IT.pvalue <= threshold] = 1

data_symbolic = np.array([x.data for x in morphed["symbolic"]])
tt_symbolic = scipy.stats.ttest_1samp(data_symbolic, popmean=0, alternative="greater")
ref_stc_data_symbolic = np.mean(data_symbolic, axis=0)
ref_stc_data_symbolic[tt_symbolic.pvalue > threshold] = 0
ref_stc_data_symbolic[tt_symbolic.pvalue <= threshold] = 1


discrete_colors = [
    (0, 0, 0, 0),                    # 0: transparent
    (209/255, 121/255, 10/255, 1),   # 2: #D1790A
]
ref_stc = morphed["symbolic"][0].copy()
ref_stc.data = ref_stc_data_symbolic
cmap = mcolors.ListedColormap(discrete_colors, name='custom_discrete_list')
symbolic = make_movie(ref_stc, cmap)

discrete_colors = [
    (0, 0, 0, 0),                    # 0: transparent
    (90/255, 43/255, 161/255, 1),    # 1: #5A2BA1
]
cmap = mcolors.ListedColormap(discrete_colors, name='custom_discrete_list')
ref_stc = morphed["IT"][0].copy()
ref_stc.data = ref_stc_data_IT
IT = make_movie(ref_stc, cmap)

discrete_colors = [
    (0, 0, 0, 0),                    # 0: transparent
    (22/255, 102/255, 41/255, 1)     # 3: #166629
]
cmap = mcolors.ListedColormap(discrete_colors, name='custom_discrete_list')
ref_stc = morphed["IT"][0].copy()
ref_stc.data = (ref_stc_data_IT + ref_stc_data_symbolic) > 1
overlap = make_movie(ref_stc, cmap)

ref_stc.data = ref_stc_data_IT * 0
empty = make_movie(ref_stc, cmap)

IT[IT == empty] = 0
symbolic[symbolic == empty] = 0
overlap[overlap == empty] = 0

empty[IT != 0] = IT[IT != 0]
empty[symbolic != 0] = symbolic[symbolic != 0]
empty[overlap != 0] = overlap[overlap != 0]

imageio.mimwrite(f"figs/merged_manually_threshold-{threshold}.mp4", empty)
