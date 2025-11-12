#!/usr/bin/env python2

import os
import numpy as np

import trajectory_utils as tu
import transformations as tf


def compute_relative_error(p_es, q_es, p_gt, q_gt, T_cm, dist, max_dist_diff,
                           accum_distances=[], rot=np.eye(3),
                           scale=1.0, unknown_gt_rot=False):

    if len(accum_distances) == 0:
        accum_distances = tu.get_distance_from_start(p_gt)
    comparisons = tu.compute_comparison_indices_length(
        accum_distances, dist, max_dist_diff)

    n_samples = len(comparisons)
    print('number of samples = {0} '.format(n_samples))
    if n_samples < 2:
        print("Too few samples! Will not compute.")
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]),\
            np.array([]), np.array([])

    T_mc = np.linalg.inv(T_cm)
    errors = []
    for idx, c in enumerate(comparisons):
        if not c == -1:
            T_c1 = tu.get_rigid_body_trafo(q_es[idx, :], p_es[idx, :])
            T_c2 = tu.get_rigid_body_trafo(q_es[c, :], p_es[c, :])
            T_c1_c2 = np.dot(np.linalg.inv(T_c1), T_c2)
            T_c1_c2[:3, 3] *= scale

            T_m1 = tu.get_rigid_body_trafo(q_gt[idx, :], p_gt[idx, :])
            T_m2 = tu.get_rigid_body_trafo(q_gt[c, :], p_gt[c, :])
            T_m1_m2 = np.dot(np.linalg.inv(T_m1), T_m2)

            T_m1_m2_in_c1 = np.dot(T_cm, np.dot(T_m1_m2, T_mc))
            T_error_in_c2 = np.dot(np.linalg.inv(T_m1_m2_in_c1), T_c1_c2)
            T_c2_rot = np.eye(4)
            T_c2_rot[0:3, 0:3] = T_c2[0:3, 0:3]
            T_error_in_w = np.dot(T_c2_rot, np.dot(
                T_error_in_c2, np.linalg.inv(T_c2_rot)))
            if unknown_gt_rot:
                # when the ground truth rotation is unknown, we assume the estimated rotations are as perfect as gt rotations.
                # but there is still a relative transform between the world frames of the two trajectories.
                T_error_in_w[:3, 3] = rot.dot(T_c2[:3, 3] - T_c1[:3, 3]) * scale - (T_m2[:3, 3] - T_m1[:3, 3])
            errors.append(T_error_in_w)

    error_trans_norm = []
    error_trans_perc = []
    error_yaw = []
    error_gravity = []
    e_rot = []
    e_rot_deg_per_m = []
    for e in errors:
        tn = np.linalg.norm(e[0:3, 3])
        error_trans_norm.append(tn)
        error_trans_perc.append(tn / dist * 100)
        ypr_angles = tf.euler_from_matrix(e, 'rzyx')
        e_rot.append(tu.compute_angle(e))
        error_yaw.append(abs(ypr_angles[0])*180.0/np.pi)
        error_gravity.append(
            np.sqrt(ypr_angles[1]**2+ypr_angles[2]**2)*180.0/np.pi)
        e_rot_deg_per_m.append(e_rot[-1] / dist)
    return errors, np.array(error_trans_norm), np.array(error_trans_perc),\
        np.array(error_yaw), np.array(error_gravity), np.array(e_rot),\
        np.array(e_rot_deg_per_m)


def compute_absolute_error(p_es_aligned, q_es_aligned, p_gt, q_gt):
    e_trans_vec = (p_gt-p_es_aligned)
    e_trans = np.sqrt(np.sum(e_trans_vec**2, 1))

    # orientation error
    e_rot = np.zeros((len(e_trans,)))
    e_ypr = np.zeros(np.shape(p_es_aligned))
    for i in range(np.shape(p_es_aligned)[0]):
        R_we = tf.matrix_from_quaternion(q_es_aligned[i, :])
        R_wg = tf.matrix_from_quaternion(q_gt[i, :])
        e_R = np.dot(R_we, np.linalg.inv(R_wg))
        e_ypr[i, :] = tf.euler_from_matrix(e_R, 'rzyx')
        e_rot[i] = np.rad2deg(np.linalg.norm(tf.logmap_so3(e_R[:3, :3])))

    # scale drift
    eps = 1e-6
    motion_gt = np.diff(p_gt, axis=0)               # shape (N-1, 3)
    motion_es = np.diff(p_es_aligned, axis=0)       # shape (N-1, 3)

    # per-step distances
    dist_gt = np.linalg.norm(motion_gt, axis=1)     # (N-1,)
    dist_es = np.linalg.norm(motion_es, axis=1)     # (N-1,)

    # debug if any zero/small GT steps
    bad = np.where(dist_gt <= eps)[0]               # indices into motion_* arrays
    if bad.size:
        print(f"[DEBUG] Found {bad.size} GT steps <= {eps} (possible duplicates / stationary frames).")
        # show up to K problematic steps with context
        K = 10
        for i in bad[:K]:
            # motion index i corresponds to pose pair (i, i+1) in p_gt
            print(f"  - step {i} -> {i+1}: dist_gt={dist_gt[i]:.9e}, dist_es={dist_es[i]:.9e}")
            print(f"    motion_gt[{i}] = {motion_gt[i]}")
            print(f"    motion_es[{i}] = {motion_es[i]}")
            print(f"    p_gt[{i}]      = {p_gt[i]}")
            print(f"    p_gt[{i+1}]    = {p_gt[i+1]}")
        if bad.size > K:
            print(f"  ... ({bad.size-K} more)")

    # safe scale error
    valid = dist_gt > eps
    e_scale_perc = np.full(dist_gt.shape, np.nan, dtype=float)
    e_scale_perc[valid] = np.abs((dist_es[valid] / dist_gt[valid] - 1.0) * 100.0)
    return e_trans, e_trans_vec, e_rot, e_ypr, e_scale_perc
