#!/usr/bin/env python
# -*- coding: utf-8 -*-

from random import sample, randint
from progress.bar import Bar

import os

import loader
import net_models
from datetime import datetime
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import rsatoolbox
import fire


def main(model="cornet_s", plot=False):
    # Let's create the "saves" folder which is needed by the nn models
    if not os.path.exists("saves"):
        os.makedirs("saves")

    num_shapes = 11

    img_db = loader.Loader()
    img_db.load_images()

    # Adapt this to use CLI args
    if "cornet" in model:
        layers = ["V4", "IT", "flatten"]
    elif "resnet" in model:
        layers = ["avgpool"]
    elif "densenet" in model:
        layers = ["norm5"]

    nn_wrapper = net_models.NetWrapper(model, img_db)
    nn_wrapper.forward_all()
    nn_wrapper.preload_all()

    for layer in layers:
        shape_names = img_db.shapes
        shape = []
        vectors = []
        splits = []

        for s1 in range(11):
            for rot in range(6):
                for dil in range(6):
                    shape.append(s1)
                    vectors.append(nn_wrapper.get((s1, rot, dil, layer)).flatten())

            split = np.concatenate([np.repeat(0, 18), np.repeat(1, 18)])
            np.random.shuffle(split)
            splits.append(split)

        data_2d = np.stack(vectors)
        splits = np.concatenate(splits)

        # Sanity check: shuffle and compare RDMs
        # np.random.shuffle(shape)

        data = rsatoolbox.data.Dataset(data_2d, obs_descriptors={"stimulus": shape, "splits": splits})
        RDM_corr = rsatoolbox.rdm.calc_rdm(data, method="crossnobis", descriptor="stimulus", cv_descriptor="splits")
        mat = RDM_corr.get_matrices()[0, :]

        df = pd.DataFrame(mat, index=shape_names, columns=shape_names)
        df.to_csv(f"output/diss_mat_model-{model}_layer-{layer}.csv")

        if plot:
            fig, ax = plt.subplots()
            im = ax.imshow(mat)

            # Show all ticks and label them with the respective list entries
            ax.set_xticks(np.arange(len(shape_names)), labels=shape_names)
            ax.set_yticks(np.arange(len(shape_names)), labels=shape_names)

            # Rotate the tick labels and set their alignment.
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

            # Loop over data dimensions and create text annotations.
            for i in range(len(shape_names)):
                for j in range(len(shape_names)):
                    text = ax.text(j, i, round(1000 * mat[i, j]), ha="center", va="center", color="w",)

            ax.set_title("1000 time dissimilarity value across")
            fig.tight_layout()
            plt.show()


if __name__ == "__main__":
    fire.Fire(main)
