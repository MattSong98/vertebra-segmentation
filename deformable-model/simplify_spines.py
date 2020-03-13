# -*- coding: utf-8 -*-
"""Using ACVD tool to simplify triangles

This script only support for ply format file. To make each triangle the same size, the number of vertices
is proportional to the area.
"""
import os
import sys
from file_importer import read_ply
from mesh_util import mesh_area

file_dir = '/Users/haoyuesong/PycharmProjects/DeformationModel/Database/data3_mc/'
output_dir = '/Users/haoyuesong/PycharmProjects/DeformationModel/Database/data4_simplified/'

if not os.path.exists(file_dir):
    print('{} not found.'.format(file_dir))
    sys.exit()
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

file_list = os.listdir(file_dir)
for i in range(1, 6):

    # collecting ith vertebra's info
    ith_vertebra_filename = 'L{}.ply'.format(i)
    ith_vertebra_mesh = read_ply(file_dir + ith_vertebra_filename)
    ith_vertebra_area = mesh_area(*ith_vertebra_mesh)
    nverts = int(ith_vertebra_area / 6.2)

    # simplifying......
    cmd1 = '/Users/haoyuesong/Downloads/VTK-7.1.1/bulid/ACVD/bin/ACVD {fname} {nverts} 0 -d 0 -o {outdir} -of {ofname}'.format(fname=file_dir + ith_vertebra_filename,
                                                                               nverts=nverts,
                                                                               outdir=output_dir, ofname=ith_vertebra_filename)
    cmd2 = 'mv {outdir}smooth_{ofname} {outdir}{ofname}'.format(outdir=output_dir, ofname=ith_vertebra_filename)
    print(cmd1+'\n'+cmd2)
    os.system(cmd1)
    os.system(cmd2)
