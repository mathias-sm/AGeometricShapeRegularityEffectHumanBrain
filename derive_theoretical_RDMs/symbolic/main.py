#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This file is the main file that highjacks a model to make it do what we want
"""

import material
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

PI = 3.14159265359

def compare_directions(direction_list, tol_angles_arg):
    """ Given a tolerance level, compare all pairs of directions
    """
    return [abs(direction_list[i] - direction_list[j]) < tol_angles_arg
            for i in range(4) for j in range(i)]


def compare_lengths(length_list, tol_arg):
    """ Given a tolerance level, compare all pairs of lengths
    """
    return [(max(length_list[i], length_list[j]) /
             min(length_list[i], length_list[j])) - 1 < tol_arg
            for i in range(4) for j in range(i)]


def compare_angles(angle_list, tol_angles_arg):
    """ Given a tolerance level, compare all pairs of angles
    """
    return [abs(angle_list[i] - angle_list[j]) < tol_angles_arg
            for i in range(4) for j in range(i)]


def compare_to_r_angle(angle_list, tol_angles_arg):
    """ Given a tolerance level, compare all to a right angle
    """
    return [abs(angle_list[i] - (PI/2)) < tol_angles_arg
            for i in range(4)]


def compute_distances(shapes, tol_arg, tol_angles_arg):
    diss_mat = {}
    for idx1, name_1 in enumerate(material.allShapes.keys()):
        diss_mat[name_1] = []
        for idx2, name_2 in enumerate(material.allShapes.keys()):
            v_ref_1 = compare_to_r_angle(shapes[name_1][0][2], tol_angles_arg)
            v_ref_1 += compare_directions(shapes[name_1][0][0], tol_angles_arg)
            v_ref_1 += compare_lengths(shapes[name_1][0][1], tol_arg)
            v_ref_1 += compare_angles(shapes[name_1][0][2], tol_angles_arg)
            v_ref_2 = compare_to_r_angle(shapes[name_2][0][2], tol_angles_arg)
            v_ref_2 += compare_directions(shapes[name_2][0][0], tol_angles_arg)
            v_ref_2 += compare_lengths(shapes[name_2][0][1], tol_arg)
            v_ref_2 += compare_angles(shapes[name_2][0][2], tol_angles_arg)
            dist = sum([int(v_ref_1[i] != v_ref_2[i]) for i in range(6+6+6+4)])
            diss_mat[name_1].append(dist / 16)  # Divide by the max, already known
        diss_mat[name_1] = np.array(diss_mat[name_1])
    diss_mat = pd.DataFrame(diss_mat)
    diss_mat.index = material.allShapes.keys()
    diss_mat.to_csv("symbolic_sym_diss_mat.csv")


if __name__ == "__main__":
    shapes = material.pre_compute()
    compute_distances(shapes, 0.125, 0.125*(PI/2))
