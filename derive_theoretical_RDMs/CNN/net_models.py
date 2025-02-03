#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This contains the psychological models that work on images
"""

import pickle
from itertools import product
import os
from progress.bar import Bar
import numpy as np
import torch
import torchvision
import cornet.cornet_s as cornet_s
import resnet.resnet as resnet
import densenet.densenet as densenet


class FeatureExtractor(torch.nn.Module):
    layer_names = None

    def inject(self, highjacked):
        self.highjacked = highjacked

    def forward(self, x):
        outputs = []
        layers = []
        for name, module in self.highjacked._modules.items():
            if name == "decoder":  # Special case for cornets
                for name2, module2 in module._modules.items():
                    x = module2(x)
                    if name2 == "flatten":
                        outputs.append(x)
                        layers.append(name2)
            elif name == "features":  # Main group for densenet
                for name2, module2 in module._modules.items():
                    x = module2(x)
                    outputs.append(x)
                    layers.append(str(name2))
            elif name == "classifier":  # Last densenet, useless
                # x = module(x)
                pass
            elif name == "fc":  # Last resnet, useless
                # x = module(x)
                pass
            else:
                x = module(x)
                outputs.append(x)
                layers.append(name)

        # WARNING : potential for race condition here. Test and report.
        if self.layer_names is None:
            self.layer_names = layers

        return outputs


class NetWrapper:
    """My wrapper around NNetwork"""

    results = {}
    currently_loaded = None
    model = None
    img_db = None
    cuda = False
    layer_names = None
    highjacked = None
    me = None

    def __init__(self, who, img_db):
        self.img_db = img_db
        self.me = who
        self.results[who] = [
            [[[{} for i in range(1)] for i in range(6)] for i in range(6)]
            for i in range(11)
        ]

        if "cornet" in who:
            self.model = cornet_s.CORnet_S()
            restore_path = None
            if who == "cornet_s":
                restore_path = "./cornet/cornet_s-1d3f7974.pth"
            elif "cornet" in who:
                restore_path = f"{who}"

            self.model = torch.nn.DataParallel(self.model)
            ckpt_data = None
            if self.cuda:
                self.model = self.model.cuda()
                ckpt_data = torch.load(restore_path)
            else:
                ckpt_data = torch.load(restore_path, map_location="cpu")

            size = ckpt_data["state_dict"]["module.decoder.linear.bias"].size()[0]
            new_last_layer = torch.nn.Linear(512, size)
            self.model.module._modules["decoder"]._modules["linear"] = new_last_layer

            self.model.load_state_dict(ckpt_data["state_dict"])

            self.model = self.model.module
        elif who == "densenet":
            self.model = densenet.densenet169()
        elif who == "resnet":
            self.model = resnet.resnet101(pretrained=False)
            restore_path = "./resnet/resnet101-5d3b4d8f.pth"
            ckpt_data = None
            if self.cuda:
                self.model = self.model.cuda()
                ckpt_data = torch.load(restore_path)
            else:
                ckpt_data = torch.load(restore_path, map_location="cpu")

            self.model.load_state_dict(ckpt_data)

        self.highjacked = FeatureExtractor()
        self.highjacked.inject(self.model)
        self.model = self.highjacked

        self.model = torch.nn.DataParallel(self.model)
        if self.cuda:
            self.model = self.model.cuda()

        self.model.eval()

    def forward_all(self):
        bar = Bar(f"Loading {self.me} processed images", max=6 * 6)

        for d, r in list(product(range(6), range(6))):
            batch = torch.stack([self.img_db.imgs[i, d, r] for i in range(11)])
            batch_size = 11

            path = f"saves/{self.me}/0/{d}_{r}.pkl"
            if not os.path.exists(path):
                output = self.model.forward(batch)
                local_results = [
                    {
                        name: output[j][i].data.cpu().numpy()
                        for j, name in enumerate(self.highjacked.layer_names)
                    }
                    for i in range(batch_size)
                ]

                for s in range(11):
                    path = f"saves/{self.me}/{s}/"
                    if not os.path.exists(path):
                        os.makedirs(path)
                    path = f"saves/{self.me}/{s}/{d}_{r}.pkl"
                    with open(path, "wb") as f:
                        p = pickle.Pickler(f)
                        p.fast = True
                        p.dump(local_results[s])

            output = None
            batch = None
            bar.next()
        bar.finish()

    def preload_all(self):
        for s in range(11):
            for d, r in list(product(range(6), range(6))):
                path = f"saves/{self.me}/{s}/{d}_{r}"
                with open(f"{path}.pkl", "rb") as f:
                    self.results[self.me][s][d][r] = pickle.load(f)

    def get(self, s1):
        (s1, d1, r1, l1) = s1
        l1 = self.results[self.me][s1][d1][r1][l1]
        return l1

    def dist(self, s1, s2):
        (s1, d1, r1, l1) = s1
        (s2, d2, r2, l2) = s2
        l1 = self.results[self.me][s1][d1][r1][l1]
        l2 = self.results[self.me][s2][d2][r2][l2]
        return np.linalg.norm(l2 - l1)

    def test(self, l, layer):
        dist_to_bar = np.empty((6))
        s, _, _, _ = l[0]
        if self.currently_loaded != s:
            self.preload_at(s)
        l = [self.results[self.me][d][r][t][layer] for (s, d, r, t) in l]
        for i in range(6):
            deprived = l[:i] + l[i + 1 :]
            b = sum(deprived) / len(deprived)
            dist_to_bar[i] = np.linalg.norm(b - l[i])
        return np.argmax(dist_to_bar) == 5


if __name__ == "__main__":
    import loader
    from random import sample

    img_db = loader.Loader()
    img_db.load_images(11)

    cornet_wrapper = NetWrapper("cornet_s", img_db)
    cornet_wrapper.forward_all()
    cornet_wrapper.preload_all()

    print("test Identity:")
    print(cornet_wrapper.dist((1, 1, 1, "IT"), (1, 1, 1, "IT")))
    print("test Symetry:")
    print(cornet_wrapper.dist((2, 1, 1, "IT"), (1, 1, 1, "IT")))
    print(cornet_wrapper.dist((1, 1, 1, "IT"), (2, 1, 1, "IT")))
