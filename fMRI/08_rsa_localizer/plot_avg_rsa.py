import fire
import pickle
import numpy as np
from nilearn.glm.contrasts import expression_to_contrast_vector, compute_contrast
from glob import glob
import rsatoolbox
import scipy
import matplotlib.pyplot as plt

import itertools
from scipy.stats import ttest_rel
from statsmodels.stats.multitest import multipletests


def significance_symbol(p):
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    return 'ns'

plot = False
data_path = "../bids_dataset/derivatives"
dist = "crossnobis"
pop = "adults"

all_average_rdms = []
for adults in ["adults", "kids"]:
    average_rdms_per_age = []
    for roi in ["rIPS", "lIPS", "rITG", "lVent", "rVent"]:
    # for roi in ["rIPS"]:
        conditions = ['shape1', 'shape3', "number", "word", 'Chinese', 'face', 'house', 'tool']

        rdms, subs = [], []
        for path in glob(f"{data_path}/rsa/sub-{2 if adults == "adults" else 3}*"):
            sub = path.split("/")[-1]
            rdms.append(pickle.load(open(f"{path}/func/{sub}_task-category_{roi}_{dist}_pop-{pop}.pkl", "rb")))

        rsa_rdms = rsatoolbox.rdm.concat(rdms)

        # Model using a "fixed eval" categorical model (not super strong?)
        geom_is_math = rsatoolbox.rdm.get_categorical_rdm([1,1,1,0,0,0,0,0], 'geom_is_math')
        model = rsatoolbox.model.ModelFixed('test', geom_is_math)
        results = rsatoolbox.inference.eval_fixed(model, rsa_rdms, method='corr_cov')
        # print(f"{roi} - {adults} - rsa stat is {scipy.stats.ttest_1samp(results.evaluations[0,0], popmean=0, alternative="greater")}")

        average_rdms_per_age.append(np.mean([rdm.get_matrices()[0] for rdm in rsa_rdms], axis=0))

        data = rsa_rdms.get_matrices()[:,2,:]
        if dist == "correlation":
            data = 1 - data

        shapes = np.mean(data[:,[0,1]], axis=1)
        notshapes = np.mean(data[:,3:], axis=1)

        if False:
            print(np.mean(np.mean(data, axis=0)[[0,1]]))
            print(np.mean(np.mean(data, axis=0)[3:]))
            shapes = np.mean(data[:,[0,1]], axis=1)
            notshapes = np.mean(data[:,3:], axis=1)
            plt.close('all')
            plt.scatter([1 for _ in range(data.shape[0])], shapes)
            plt.scatter([2 for _ in range(data.shape[0])], notshapes)
            plt.scatter(1, np.mean(shapes))
            plt.scatter(2, np.mean(notshapes))
            plt.savefig("figs/test.svg")

        stat2 = scipy.stats.ttest_1samp(shapes - notshapes, popmean=0, alternative="less")
        print(f"{roi} - {adults} - pair stat is {stat2.pvalue}")

        # plot:
        if plot:
            plt.close('all')
            plt.imshow(average_rdms_per_age[-1], interpolation="none")
            ax = plt.gca()
            ax.set_xticks(np.arange(8))
            ax.set_yticks(np.arange(8))
            ax.set_xticklabels(conditions)
            ax.set_yticklabels(conditions)
            plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")
            plt.title(f"{adults} - {roi}")
            plt.savefig(f"figs/avg_rdm_roi-{roi}-{adults}_{dist}.svg") 

            data = rsa_rdms.get_matrices()[:,2,:]
            if dist == "correlation":
                data = 1 - data
            means = data.mean(axis=0)
            sem   = data.std(axis=0, ddof=1) / np.sqrt(data.shape[0])

            plt.close('all')
            fig, ax = plt.subplots(figsize=(8, 6))
            x = np.arange(data.shape[1])
            bars = ax.bar(x, means, yerr=sem, capsize=5)

            pairs = list(itertools.combinations(range(0,data.shape[1]), 2))
            pvals = [ttest_rel(data[:, i], data[:, j]).pvalue for i, j in pairs]

            reject, pvals_corr, _, _ = multipletests(pvals, alpha=0.05, method='holm')

            y_max = (means + sem).max()
            step = y_max * 0.05
            current_y = y_max + step

            for (pair, sig, p_adj) in zip(pairs, reject, pvals_corr):
                if sig:  # mark only significant comparisons
                    i, j = pair
                    ax.plot([i, i, j, j],
                            [current_y, current_y + step*0.25,
                             current_y + step*0.25, current_y],
                            linewidth=1)
                    ax.text((i + j) / 2, current_y + step*0.25,
                            significance_symbol(p_adj),
                            ha='center', va='bottom')
                    current_y += step * 1.5  # shift up for next annotation

            ax.set_xlabel('Condition')
            ax.set_ylabel('Mean ± SEM')
            ax.set_xticks(x)
            ax.set_title('Barplot with SEM and Pairwise Paired t-test Significance')
            ax.set_xticks(np.arange(8)) 
            ax.set_xticklabels(conditions)
            plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

            fig.tight_layout()
            plt.title(f"{adults} - {roi} (barplot against shape1)")
            fig.savefig(f"figs/avg_rdm_roi-{roi}-{adults}_barplot_{dist}.svg")

    all_average_rdms.append(np.array(average_rdms_per_age))
