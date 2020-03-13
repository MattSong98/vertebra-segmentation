# -*- coding: utf-8 -*-
"""Module to save mesh files
"""
import numpy as np
import struct
from stl import mesh


def write_obj(filename, verts, faces):
    with open(filename, 'w') as f:
        # f.write("# OBJ file\n")
        for v in verts:
            f.write("v {} {} {}\n".format(*v))
        for p in faces:
            f.write("f {} {} {}\n".format(*(p + 1)))


def write_off(filename, verts, faces):
    with open(filename, 'w') as f:
        f.write("OFF\n")
        f.write("{} {}\n".format(verts.shape[0], faces.shape[0]))
        for v in verts:
            f.write("{} {} {}\n".format(*v))
        for p in faces:
            f.write("{} {} {} {}\n".format(3, *p))


def write_stl(filename, verts, faces):
    cube = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            cube.vectors[i][j] = verts[f[j], :]
    cube.save(filename)


def write_ply_ascii(filename, verts, faces, normal=False):
    nverts = verts.shape[0]
    if faces is None:
        nfaces = 0
        faces = []
    else:
        nfaces = faces.shape[0]

    with open(filename, 'w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write('element vertex {}\n'.format(nverts))
        f.write('property float32 x\n')
        f.write('property float32 y\n')
        f.write('property float32 z\n')
        if normal:
            f.write('property float32 nx\n')
            f.write('property float32 ny\n')
            f.write('property float32 nz\n')
        f.write('element face {}\n'.format(nfaces))
        f.write('property list uint8 int32 vertex_indices\n')
        f.write('end_header\n')
        for v in verts:
            if normal:
                f.write("{} {} {} {} {} {}\n".format(*v))
            else:
                f.write("{} {} {}\n".format(*v))
        for p in faces:
            f.write("{} {} {} {}\n".format(3, *p))


def write_ply_binary(filename, verts, faces, normal=False):
    nverts = verts.shape[0]
    if faces is None:
        nfaces = 0
        faces = []
    else:
        nfaces = faces.shape[0]

    with open(filename, 'wb') as f:
        f.write(b'ply\n')
        f.write(b'format binary_little_endian 1.0\n')
        f.write('element vertex {}\n'.format(nverts).encode())
        f.write(b'property float x\n')
        f.write(b'property float y\n')
        f.write(b'property float z\n')
        if normal:
            f.write(b'property float32 nx\n')
            f.write(b'property float32 ny\n')
            f.write(b'property float32 nz\n')
        f.write('element face {}\n'.format(nfaces).encode())
        f.write(b'property list uchar int vertex_indices\n')
        f.write(b'end_header\n')
        for v in verts:
            if normal:
                f.write(struct.pack('<ffffff', *v))
            else:
                f.write(struct.pack('<fff', *v))
        for p in faces:
            f.write(struct.pack('<Biii', 3, *p))


def write_ply(filename, verts, faces=None, asc=False, normal=False):
    """Give vertices and faces, write ply file

    Args:
        filename (str): ply file name
        verts (ndarray): vertices, shape (M,3) or (M,6)
        faces (ndarray): faces, shape (N,3), if None, means only write vertices
        asc (bool): if True, use ascii format, else use binary format
        normal (bool): if you want to save normal information, check this option and
            concat normals to verts, which makes verts a (M,6) ndarray
    """
    if asc:
        write_ply_ascii(filename=filename, verts=verts, faces=faces, normal=normal)
    else:
        write_ply_binary(filename=filename, verts=verts, faces=faces, normal=normal)
