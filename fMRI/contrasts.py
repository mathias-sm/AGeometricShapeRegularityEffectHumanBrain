import numpy as np

shapes = {}
shapes["easy"] = ["square", "rectangle", "isoTrapezoid", "losange", "hinge", "random"]
shapes["hard"] = ["square", "rectangle", "isoTrapezoid", "parallelogram", "losange", "kite", "hinge", "random"]

geom_preds = {
    "theory": {
        "square": 18,
        "rectangle": 14,
        "isoTrapezoid": 7,
        "parallelogram": 7,
        "losange": 11,
        "kite": 3,
        "rightkite": 5,
        "rustedhinge": 8,
        "hinge": 7,
        "trapezoid": 5,
        "random": 2,
    },
    "behavior_online": {
        "square": 0.0747863247863248,
        "rectangle": 0.0651709401709402,
        "isoTrapezoid": 0.150793650793651,
        "parallelogram": 0.178266178266178,
        "losange": 0.214285714285714,
        "kite": 0.230006105006105,
        "rightKite": 0.256715506715507,
        "rustedHinge": 0.284696784696785,
        "hinge": 0.302808302808303,
        "trapezoid": 0.357244607244607,
        "random": 0.421560846560847,
    },
    "behavior_scanner": { # Aggregated on all subjects in both age groups
        "square": 0.0880462775528565,
        "rectangle": 0.0951537899235268,
        "isoTrapezoid": 0.236390148164529,
        "parallelogram": 0.336769005847953,
        "losange": 0.239995571862348,
        "kite": 0.426464076858814,
        "hinge": 0.266724459858167,
        "random": 0.317654306746315,
    }
}


contrast_motor    = "star"
contrast_ffa      = "3*face              - (word + tool + house)"
contrast_vwfa     = "3*word              - (face + tool + house)"
contrast_tool     = "3*tool              - (face + word + house)"
contrast_house    = "3*house             - (face + word + tool)"
contrast_shapes   = "2*shape1 + 2*shape3 - (face + tool + house + word)"
contrast_shape1   = "3*shape1            - (face + tool + house)"
contrast_notshape = "3*star              - (face + tool + house)"
contrast_shape3   = "shape3              - word"
contrast_shape1b  = "4*shape1            - (face + tool + house + Chinese)"
contrast_shape3b  = "2*shape3            - (word + number)"

def make_contrast(d, shapes, coef=1):
    vals = np.array([d[x] for x in shapes])
    normed_vals = coef * (vals - np.mean(vals)) / np.std(vals)
    return(" + ".join([f"({normed_vals[i]} * {s})" for i, s in enumerate(shapes)]))

geom_contrasts = {}
for shapes_cond in ["easy", "hard"]:
    for pred, coef in zip(["theory", "behavior_online", "behavior_scanner"], [-1,1,1]):
        geom_contrasts[(pred, shapes_cond)] = make_contrast(geom_preds[pred], shapes[shapes_cond], coef)

contrasts = dict()
contrasts["category"] = {
    "motor": contrast_motor,
    "ffa": contrast_ffa,
    "vwfa": contrast_vwfa,
    "tool": contrast_tool,
    "house": contrast_house,
    "all_shapes": contrast_shapes,
    "shape1": contrast_shape1,
    "shape3": contrast_shape3,
    "shape1b": contrast_shape1b,
    "shape3b": contrast_shape3b,
    "notshape": contrast_notshape,
}

contrasts["geometry"] = {
    "geom_theory": geom_contrasts[('theory','easy')],
    "geom_behavior_online": geom_contrasts[('behavior_online','easy')],
    "geom_behavior_scanner": geom_contrasts[('behavior_scanner','easy')],
    "any": "0.166666*(rectangle + square + isoTrapezoid + losange + hinge + random)"
}

contrasts["geometryHard"] = {
    "geom_theory": geom_contrasts[('theory','hard')],
    "geom_behavior_online": geom_contrasts[('behavior_online','hard')],
    "geom_behavior_scanner": geom_contrasts[('behavior_scanner','hard')],
    "colors_grad": "color_6 - color_2",
    "any": "0.125*(rectangle + square + isoTrapezoid + parallelogram + kite + losange + hinge + random)"
}
