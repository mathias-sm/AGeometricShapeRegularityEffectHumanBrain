import numpy as np
import pickle
import fire
from matplotlib.colors import LinearSegmentedColormap
import scipy

based = "../../bids_data/derivatives"
basemsm = f"{based}/msm/sub-average/meg/sub-average_task-POGS_"

def make_movie(stc, name):
    pos_max = np.max(stc.data)
    colormap = "hot"
    if name == "symbolic":
        cdict = {'red':   [[0.0, 241/255, 241/255], [1.0, 241/255, 241/255]],
                 'green': [[0.0, 163/255, 163/255], [1.0, 163/255, 163/255]],
                 'blue':  [[0.0, 64/255, 64/255], [1.0, 64/255, 64/255]],
                 'alpha': [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]}
        colormap = LinearSegmentedColormap('testCmap', segmentdata=cdict, N=256)
    elif name == "IT":
        cdict = {'red':   [[0.0, 153/255, 153/255], [1.0, 153/255, 153/255]],
                 'green': [[0.0, 142/255, 142/255], [1.0, 142/255, 142/255]],
                 'blue':  [[0.0, 195/255, 195/255], [1.0, 195/255, 195/255]],
                 'alpha': [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]}
        colormap = LinearSegmentedColormap('testCmap', segmentdata=cdict, N=256)
    brain = stc.plot(
            hemi="split",
            colormap = colormap,
            views=["lateral", "medial"],
            subjects_dir=f"{based}/freesurfer/subjects/",
            size=(1920, 1080),
            smoothing_steps=4,
            surface="white",
            clim=dict(kind="value", lims=[0,pos_max/2,pos_max]),
            show_traces=False,
            time_viewer=False,
            # colorbar=False,
            brain_kwargs=dict(offscreen=True, show=False))
    brain.save_movie(f"figs/{name}.mp4", time_dilation=40, interpolation="cubic", framerate=25)
    brain.close()


morphed = pickle.load(open(f"{based}/msm/sub-average/meg/sub-average_task-POGS_all+rsa+many.pkl", "rb"))

for name in ["symbolic", "IT"]:
    data = np.array([x.data for x in morphed[name]])
    tt = scipy.stats.ttest_1samp(data, popmean=0, alternative="greater")
    ref_stc_data = np.mean(data, axis=0)
    ref_stc_data[tt.pvalue > .01] = 0
    ref_stc = morphed[name][0].copy()
    ref_stc.data = ref_stc_data
    ref_stc.crop(0, 0.6)
    make_movie(ref_stc, name)
