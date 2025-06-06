import numpy as np
import rsatoolbox
import scipy.stats
import matplotlib.pyplot as plt
import met_brewer
from matplotlib.colors import LinearSegmentedColormap

cm = "crossnobis"
smooth = "unsmooth"
rdms_movie = rsatoolbox.rdm.rdms.load_rdm(f"./all_rdms/rdms_{cm}_{smooth}.pkl")
for time in [0.084, 0.232]:
    topred = rdms_movie.subset('time', [time])

    toplot = topred.get_matrices()
    for i in range(20):
        toplot[i] = scipy.stats.zscore(toplot[i])

    avg_toplot = np.mean(toplot, axis=0)
    avg_toplot = (avg_toplot + avg_toplot.T) / 2
    avg_toplot[np.triu_indices(avg_toplot.shape[0], 0)] = np.nan

    plt.close('all')
    colors = met_brewer.met_brew("Hiroshige", n=100, brew_type="continuous")
    cmap = LinearSegmentedColormap.from_list('my_cmap', colors)
    plt.imshow(avg_toplot, cmap=cmap, origin="lower")
    plt.savefig(f"figs/rdm_at_t-{time}ms_{smooth}.svg", bbox_inches='tight')
