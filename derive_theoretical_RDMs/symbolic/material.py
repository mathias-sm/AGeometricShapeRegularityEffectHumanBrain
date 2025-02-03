#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import numpy as np

allShapes = {
    "square"        : [[1.2602,0],[0,1.2602],      [1.2602,1.2602],[0.6301,0.6301]], # noqa
    "rectangle"     : [[1.5,0],   [0,1],           [1.5,1],        [0.75,0.5]],      # noqa
    "isoTrapezoid"  : [[0.7445,0],[-0.3648,1.3617],[1.1351,1.3617],[0.3786,0.6808]], # noqa
    "parallelogram" : [[1.5,0],   [0.5172,0.8958], [2.0172,0.8958],[1.0086,0.4479]], # noqa
    "losange"       : [[1.3,0],   [0.9076,0.9306], [2.2076,0.9306],[1.1038,0.4653]], # noqa
    "kite"          : [[1.0427,0],[-0.3882,1.4488],[1.0557,1.0426],[0.4275,0.6228]], # noqa
    "rightKite"     : [[1.0376,0],[0,1.5],         [1.4035,0.9709],[0.6103,0.6177]], # noqa
    "rustedHinge"   : [[1.5,0],   [-0.1021,0.5790],[1.1266,1.4394],[0.6311,0.5046]], # noqa
    "hinge"         : [[1.5,0],   [0,0.7],         [1.3594,1.3339],[0.7148,0.5084]], # noqa
    "trapezoid"     : [[0.951,0], [0.227,1.2],     [1.727,1.2],    [0.7262,0.6]],    # noqa
    "random"        : [[0.7,0],   [0.1604,1.1387], [1.6092,1.5269],[0.6174,0.6664]], # noqa
}


def rotate(v, theta):
    theta = theta * (2 * math.pi) / 360

    return [v[0] * math.cos(theta) - v[1] * math.sin(theta),
            v[0] * math.sin(theta) + v[1] * math.cos(theta)]


def angle2d(v1, v2):
    raw = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    rounded = round(raw, 15)
    return math.acos(rounded)


def angle_ref(v1):
    return angle2d(v1, np.array([1, 0]))


def observed_values(shape):
    v_ab = shape[0]
    v_aap = shape[1]
    v_abp = shape[2]
    v_apbp = v_abp - v_aap
    v_bbp = v_abp - v_ab

    # d stands for distance
    d_ab = angle_ref(v_ab)
    d_aap = angle_ref(v_aap)
    d_apbp = angle_ref(v_apbp)
    d_bbp = angle_ref(v_bbp)

    # l stands for length
    l_ab = np.linalg.norm(v_ab)
    l_aap = np.linalg.norm(v_aap)
    l_apbp = np.linalg.norm(v_apbp)
    l_bbp = np.linalg.norm(v_bbp)

    angle_apab = angle2d(-v_aap, v_ab)
    angle_aapbp = angle2d(v_apbp, -v_aap)
    angle_apbpb = angle2d(-v_bbp, -v_apbp)
    angle_bpba = angle2d(-v_ab, v_bbp)
    return [[d_ab, d_aap, d_bbp, d_apbp],
            [l_ab, l_aap, l_bbp, l_apbp],
            [angle_apab, angle_aapbp, angle_apbpb, angle_bpba]]


def pre_compute():
    observable_data = dict()
    for name, points in allShapes.items():
        points = np.array(points[0:3]).copy()
        observable_data[name] = [observed_values(points)]
        for i in range(4):
            lpoints = points.copy()
            observable_data[name] += [observed_values(lpoints).copy()]
    return observable_data
