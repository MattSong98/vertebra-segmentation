"""Utilities to manipulate mesh data

A mesh data consists of vertices and faces
"""
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import connected_components
from numba import njit


def mesh2graph(verts, faces):
    """Build graph from mesh

    Returns:
        graph: a csr format sparse matrice with shape (nverts, nverts),
            True means connection between two vertices
    """
    nverts = verts.shape[0]
    nfaces = faces.shape[0]
    graph = lil_matrix((nverts, nverts), dtype=bool)
    for i in range(nverts):
        graph[i, i] = True
    for i in range(nfaces):
        t = faces[i]
        graph[t[0], t[1]] = True
        graph[t[1], t[2]] = True
        graph[t[2], t[0]] = True
        graph[t[1], t[0]] = True
        graph[t[2], t[1]] = True
        graph[t[0], t[2]] = True
    graph = graph.tocsr()
    return graph


def vvLUT(verts, faces):
    nverts = verts.shape[0]
    nfaces = faces.shape[0]
    LUT = {i: set() for i in range(nverts)}
    for i in range(nfaces):
        t = faces[i]
        LUT[t[0]].add(t[1])
        LUT[t[1]].add(t[2])
        LUT[t[2]].add(t[0])

    # sysmmetric assert
    for key in LUT.keys():
        linkto = LUT[key]
        for l in linkto:
            if key not in LUT[l]:
                raise Exception("vvLUT not symmetric!")

    return LUT


def vfLUT(verts, faces):
    """Build a LUT for finding connected faces giving vertices

    Returns:
        LUT (dict): {vertid1: {facesid1, faceid2, ...}, ...}
    """
    nverts = verts.shape[0]
    nfaces = faces.shape[0]
    LUT = {i: set() for i in range(nverts)}
    for i in range(nfaces):
        t = faces[i]
        LUT[t[0]].add(i)
        LUT[t[1]].add(i)
        LUT[t[2]].add(i)
    return LUT


def remove_isolate_components(verts, faces):
    vflut = vfLUT(verts, faces)
    graph = mesh2graph(verts, faces)
    ncomp, labeled_verts = connected_components(graph)
    argmax_index = [(labeled_verts == i).astype(np.int).sum() for i in range(ncomp)]
    argmax_index = np.array(argmax_index).argmax()
    vmask = np.where(labeled_verts == argmax_index)[0]
    sverts = verts[vmask]
    vfluts = [vflut[i] for i in vmask]
    sfaces = set()
    for i in range(len(vfluts)):
        sfaces |= vfluts[i]
    sfaces = faces[list(sfaces)]
    index_fix = {v: i for i, v in enumerate(vmask)}
    index_fix_func = np.vectorize(lambda x: index_fix[x])
    sfaces = index_fix_func(sfaces)
    return sverts, sfaces


@njit(cache=True)
def face_normal(verts, faces):
    """Return normals of faces

    The norm of normals is proportion to the area of neighbour faces
    """
    nfaces = faces.shape[0]
    normals = np.zeros((nfaces, 3))
    for i in range(nfaces):
        ai, bi, ci = faces[i]
        a, b, c = verts[ai], verts[bi], verts[ci]
        u = b - a
        v = c - a
        normals[i, 0] = u[1] * v[2] - u[2] * v[1]
        normals[i, 1] = u[2] * v[0] - u[0] * v[2]
        normals[i, 2] = u[0] * v[1] - u[1] * v[0]

    return normals


@njit(cache=True)
def face_center(verts, faces):
    nfaces = faces.shape[0]
    centers = np.zeros((nfaces, 3))
    for i in range(nfaces):
        ai, bi, ci = faces[i]
        a = verts[ai]
        b = verts[bi]
        c = verts[ci]
        centers[i] = (a + b + c) / 3

    return centers


def vertex_normal(verts, faces):
    nverts = verts.shape[0]
    vflut = vfLUT(verts, faces)
    face_normals = face_normal(verts, faces)
    normals = np.zeros((nverts, 3))
    for i in range(nverts):
        normals[i] = face_normals[list(vflut[i])].sum(axis=0)
    return normals


def elementwise_normalize(points):
    return points / np.sqrt(np.sum(points ** 2, axis=1) + 1e-5)[:, np.newaxis]


def mesh_area(verts, faces):
    normals = face_normal(verts, faces)
    areas = np.sqrt(np.sum(normals ** 2, axis=1))
    return areas.sum()


def inverse_axis(verts):
    return np.c_[verts[:, 2], verts[:, 1], verts[:, 0]]


def rotation_matrix(theta, axis='x'):
    s = np.sin(theta)
    c = np.cos(theta)
    if axis == 'x':
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == 'y':
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    if axis == 'z':
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def euler_matrix(a, b, c):
    ca = np.cos(a)
    sa = np.sin(a)
    cb = np.cos(b)
    sb = np.sin(b)
    cc = np.cos(c)
    sc = np.sin(c)
    return np.array([[ca * cb * cc - sa * sc, -cc * sa - ca * cb * sc, ca * sb],
                     [cb * cc * sa + ca * sc, ca * cc - cb * sa * sc, sa * sb],
                     [-cc * sb, sb * sc, cb]])


