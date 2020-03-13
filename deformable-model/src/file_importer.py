# -*- coding: utf-8 -*-
"""Module to read mesh files

Now only support binary and ascii 'ply' format file
"""
import numpy as np
import struct
import nibabel as nib


def read_nii(filename, normalize=False):
    image = nib.load(filename)
    image_data = image.get_data()
    image_spacing = image.header.get_zooms()
    if normalize:
        dmin, dmax = image_data.min(), image_data.max()
        image_data = ((image_data - dmin) / (dmax - dmin)).astype(np.float32)
    return image_data, image_spacing



def read_ply(filename, normal=True):
    """ply format file reader

    Args:
        filename (str): ply file name
        normal (bool): if True and the file saves normals, concat normals to V,
            if False, only return vertices in V

    Returns:
        V (ndarray): vertices, shape (M, 3)
        F (ndarray): faces, shape (N, 3)
    """
    with open(filename, "rb") as f:
        nface = 0
        nverts = 0
        vert_dim = 3
        while True:
            curline = f.readline()
            if b'ascii' in curline:
                return read_ply_ascii(filename, normal=normal)
            if (b'end_header' in curline) or (b'end header' in curline):
                break
            if (b'element face' in curline) or (b'element face' in curline):
                nface = int(curline.split()[2])
            if (b'element vertex' in curline) or (b'element vertex' in curline):
                nverts = int(curline.split()[2])
            if (b'float nx' in curline) or (b'float32 nx' in curline):
                vert_dim = 6
        V = np.zeros((nverts, vert_dim), dtype=np.float32)
        F = np.zeros((nface, 3), dtype=np.uint32)
        for i in range(nverts):
            V[i, :] = struct.unpack('<' + 'f' * vert_dim, f.read(vert_dim * 4))
        for i in range(nface):
            F[i, :] = struct.unpack("<Biii", f.read(1 + 3 * 4))[1:4]
    if normal:
        return V, F
    else:
        return V[:, :3], F


def read_ply_ascii(filename, normal=True):
    """ply file reader for ascii format

    Don't use this function if you are not sure the file is saved in ascii format.
    Please use 'read_ply' anyway, because 'read_ply' can determine whether ascii or binary
    """
    with open(filename, "r") as f:
        nface = 0
        nverts = 0
        vert_dim = 3
        while True:
            curline = f.readline()
            if ('end_header' in curline) or ('end header' in curline):
                break
            if ('element face' in curline) or ('element face' in curline):
                nface = int(curline.split()[2])
            if ('element vertex' in curline) or ('element vertex' in curline):
                nverts = int(curline.split()[2])
            if ('float nx' in curline) or ('float32 nx' in curline):
                vert_dim = 6
        V = np.zeros((nverts, vert_dim), dtype=np.float32)
        F = np.zeros((nface, 3), dtype=np.uint32)
        for i in range(nverts):
            V[i, :] = [float(v) for v in f.readline().strip().split(' ')]
        for i in range(nface):
            F[i, :] = [int(v) for v in f.readline().strip().split(' ')[1:]]
    if normal:
        return V, F
    else:
        return V[:, :3], F

