# -*- coding: utf-8 -*-
"""Simple script to remove isolate components of mesh data
"""
import os
import sys
from mesh_util import remove_isolate_components
from file_importer import read_ply
from file_exporter import write_ply

file_dir = '/Users/haoyuesong/PycharmProjects/DeformationModel/Database/data3_mc/'
output_dir = '/Users/haoyuesong/PycharmProjects/DeformationModel/Database/data3_mc/'

if not os.path.exists(file_dir):
    print('{} not found.'.format(file_dir))
    sys.exit()
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

for fname in os.listdir(file_dir):
    V, F = read_ply(file_dir + fname)
    SV, SF = remove_isolate_components(V, F)
    write_ply(output_dir + fname, SV, SF)
    print('{}: remove {} verts and {} faces'.format(fname, V.shape[0] - SV.shape[0], F.shape[0] - SF.shape[0]))
