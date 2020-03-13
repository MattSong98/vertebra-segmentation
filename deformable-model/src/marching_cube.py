# -*- coding: utf-8 -*-
"""Performing marching cubes to data_prediction: subject 1
"""
import nibabel as nib
from skimage import measure
from file_exporter import write_ply

""" 怎么保证ply记录下了CT真实的xyz比例？
1. image.get_data()
2. image.header.get_zooms()
"""
def marching_vertebra(img, spacing, label):
    # label = [1,2,3,4,5], which stand for L1-L5 respectively
    spine = list()
    for value in label:
        print('processing L{}'.format(value))
        tmp = (img == value).astype('H')
        try:
            verts, faces, _, _ = measure.marching_cubes_lewiner(tmp, 0.5)
            verts = spacing * verts
        except ValueError:
            verts, faces = None, None
        spine.append([verts,faces])
    return spine

subject = '/Users/haoyuesong/PycharmProjects/DeformationModel/Database/data1_prediction/subject2/truth.nii.gz'
output_dir = '/Users/haoyuesong/PycharmProjects/DeformationModel/Database/data3_mc'

# get volumetric data of subject 1
label = [1, 2, 3, 4, 5]
image = nib.load(subject)
img = image.get_data()
spacing = image.header.get_zooms()
spine = marching_vertebra(img, spacing, label)

# marching subject1 and save each vertebra as "L{}.ply"
for index, vertebra in enumerate(spine, start=1):
    if vertebra is not None:
        write_ply('{}/L{}.ply'.format(output_dir, index), vertebra[0], vertebra[1])
