import numpy as np
import pickle
import fire
from matplotlib.colors import LinearSegmentedColormap
import scipy

based = "../../bids_data/derivatives"
basemsm = f"{based}/msm/sub-average/meg/sub-average_task-POGS_"

def make_movie(stc, name):
    brain = stc.plot(
            hemi="split",
            views=["lateral", "medial"],
            subjects_dir=f"{based}/freesurfer/subjects/",
            size=(1920, 1080),
            smoothing_steps=4,
            surface="white",
            # show_traces=False,
            # time_viewer=False,
            colorbar=False,
            brain_kwargs=dict(offscreen=True, show=False))
    brain.save_movie(f"figs/{name}.mp4", time_dilation=40, interpolation="cubic", framerate=25)
    brain.close()


morphed = pickle.load(open(f"{based}/msm/sub-average/meg/sub-average_task-POGS_all+rsa+many.pkl", "rb"))

data_symbolic = np.array([x.data for x in morphed["symbolic"]])
data_dino = np.array([x.data for x in morphed["dino"]])

data_contrast = data_symbolic - data_dino

src = f"{based}/freesurfer/subjects/fsaverage/bem/fsaverage-ico-5-src.fif"
from mne import read_source_spaces, spatial_src_adjacency, set_log_file
src = read_source_spaces(src)
adjacency = spatial_src_adjacency(src)

import mne

clu = mne.stats.spatio_temporal_cluster_1samp_test(
    np.transpose(data_contrast, axes=(0,2,1)),
    adjacency=adjacency,
    threshold = dict(start=0, step=0.3),
    check_disjoint=True,
    tail=1,
    n_permutations=2**8,
    n_jobs=6,
    verbose=True,
)

_, cl, clp, _ = clu
np.min(clp)


tt = scipy.stats.ttest_1samp(data_symbolic - data_dino, popmean=0)

ref_stc_data = tt.statistic
ref_stc_data[tt.pvalue > .001] = 0
ref_stc = morphed["symbolic"][0].copy()
ref_stc.data = ref_stc_data
ref_stc.crop(0, 0.6)
make_movie(ref_stc, "contrast")
