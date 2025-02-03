#!/usr/bin/env python
# -*- coding: utf-8 -*-

from itertools import product
from PIL import Image
from progress.bar import Bar
from torchvision import transforms


class Loader:
    shapes = [
        "square",
        "rectangle",
        "isoTrapezoid",
        "parallelogram",
        "losange",
        "kite",
        "rightKite",
        "rustedHinge",
        "hinge",
        "trapezoid",
        "random",
    ]
    types = ["reference"]
    imgs = dict()

    transform = transforms.Compose([transforms.ToTensor()])

    def get_shapes(self):
        return self.shapes

    def load_images(self):
        max_bar = 11 * 6 * 6
        bar = Bar(f"Preloading images", max=max_bar)
        for s, d, r in list(product(range(11), range(6), range(6))):
            fname = None
            fname = f"stimuli/geom_{self.shapes[s]}_{self.types[0]}_{1+d}_{1+r}.png"
            bar.next()
            if fname is not None:
                im = Image.open(fname).convert("RGB")
                im = self.transform(im)
                self.imgs[s, d, r] = im
            else:
                self.imgs[s, d, r] = None
        bar.finish()
