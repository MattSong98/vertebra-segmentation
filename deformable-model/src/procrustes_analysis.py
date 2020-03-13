# -*- coding: utf-8 -*-
"""Procrustes analysis tool functions

including GPA (general procrustes analysis) and WEOP (Weighted Extended Orthongoal Procrustes Analysis)

ref: "Generalized Procrustes analysis and its applications in photogrammetry" by Akca, Devrim 2003
"""
import numpy as np


def gpa(datas, masks, tol=1e-2, maxiteration=100):
    """General Procrustes Analysis

    Finding geometrical centroid of more than two points matrices

    Args:
        datas (list of ndarray): each ndarray is a model points matrice.
            All matrices should have the same dimensions, say (M, 3)
        masks (list of ndarray): boolean mask matrices correspond to datas.
            1 for exist, 0 for missing, each ndarray's shape is (M, 1)
        tol (float): stop iteration if sum of square of new centroid is smaller than tol
        maxiteration (int): stop iteration condition

    Returns:
        cent (ndarray): geometrical centroid have the same dimensions of data, say (M, 3)
        transforms (list of list): each list saves the scale, rotation, translation info of the corresponding data
    """
    ndata = len(datas)
    M = datas[0].shape[0]
    masks = [np.array(m).reshape((M, 1)) for m in masks]
    smask = np.stack((m for m in masks)).sum(axis=0)
    smask = 1 / smask
    transforms = [[] for i in range(ndata)]

    # initialize cent
    cent = datas[0]
    for i in range(maxiteration):
        edatas = []
        for j in range(ndata):
            # solve WEOP
            weight = masks[j]
            s, R, t = weop(datas[j], cent, weight)
            edatas.append(s * datas[j].dot(R) + t.T)
            transforms[j] = [s, R, t]
        # update cent
        sedatas = np.stack((edatas[i] * masks[i] for i in range(len(edatas)))).sum(axis=0)
        new_cent = sedatas * smask
        error = np.sum((new_cent - cent) ** 2)
        # print('error: {}'.format(error))
        if error < tol:
            break
        cent = new_cent

    return cent, transforms


def weop(A, B, Q=1):
    """Weighted Extended Orthongoal Procrustes Analysis

    Estimate the scale, rotation and translation from A to B with weight Q

    Args:
        A (ndarray): point cloud A, shape (M, N)
        B (ndarray): point cloud B, shape (M, N)
        Q (ndarray): weight, shape (M, 1)

    Returns:
        s (float): scale
        R (ndarray): rotation matrice, shape (N,N)
        t (ndarray): translation, shape (N,1)
    """
    if (isinstance(Q, int) or isinstance(Q, float)) and Q == 1:
        J = np.ones((A.shape[0], 1))
        nvis = A.shape[0]
        temp = A.T * J.T * (1 - 1 / nvis)
        temp1 = temp.dot(B)
        temp2 = temp.dot(A)
        V, D, Wh = np.linalg.svd(temp1)
        R = V.dot(Wh)
        s = np.trace(R.T.dot(temp1)) / np.trace(temp2)
        t = np.sum(B - s * A.dot(R), axis=0) / nvis
        return s, R, t

    elif Q.any():
        J = np.ones((A.shape[0], 1))
        Aw = A * Q
        Bw = B * Q
        Jw = J * Q
        nvis = np.sum(Jw * Jw)
        temp = Aw.T * (J.T - Jw.T * Jw.T / nvis)
        temp1 = temp.dot(Bw)
        temp2 = temp.dot(Aw)
        V, D, Wh = np.linalg.svd(temp1)
        R = V.dot(Wh)
        s = np.trace(R.T.dot(temp1)) / np.trace(temp2)
        t = (Bw - s * Aw.dot(R)).T.dot(Jw) / nvis
        return s, R, t

    else:
        dim = A.shape[1]
        return 0, np.zeros((dim, dim)), np.zeros((dim, 1))
