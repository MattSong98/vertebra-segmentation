import matplotlib.pyplot as plt
import numpy as np
from segment_util import VertebraSegmentation
from file_importer import read_ply, read_nii

img, spacing = read_nii('/Users/haoyuesong/PycharmProjects/DeformationModel/Database/data1_prediction/subject1/data_original.nii.gz', normalize=True) # normalize = True CT值标准化
segfunc = VertebraSegmentation(img, spacing)


"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"""
# Actually, normal = False
verts, faces = read_ply('../data3_mean/v17.ply')
t1 = np.array([355, 183, 287])*spacing
t2 = np.array([0.83333333, 15.55555556, -3.33333333])
t3 = np.array([0.58333333, 1.75, 0.])
s = 0.9833333333333334
R = np.array([[0.99717598, -0.01923139, -0.07259629],
              [0.01950002, 0.99980538, 0.00299333],
              [0.07252459, -0.0044005, 0.99735692]])
verts -= verts.mean(axis=0)      #中心为原点的腰五的平均模型mesh
verts = s*verts.dot(R) + t1 + t2 + t3   #将mesh置于CT坐标系中进行变换
mesh = segfunc.segment((verts, faces), 17, max_search_radius=15, iteration=20)   # 分割
"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"""


def show_in_3axis(ax1, ax2, ax3, img, spacing, verts, color):
  projverts = proj2plane(verts, spacing, 255)
  ax1.scatter(projverts[:, 1], projverts[:, 0], marker='.', c=color)
  projverts = proj2plane(verts, spacing, 255 - 20)
  ax2.scatter(projverts[:, 1], projverts[:, 0], marker='.', c=color)
  projverts = proj2plane(verts, spacing, 255 + 20)
  ax3.scatter(projverts[:, 1], projverts[:, 0], marker='.', c=color)

  ax1.imshow(img[:, :, 255], cmap='gray')
  ax2.imshow(img[:, :, 255 - 20], cmap='gray')
  ax3.imshow(img[:, :, 255 + 20], cmap='gray')
  ax1.set_aspect(3)
  ax2.set_aspect(3)
  ax3.set_aspect(3)


def proj2plane(pts, spacing, value):
  a = pts[:, 2]/spacing[2] < (value + 2)
  b = pts[:, 2]/spacing[2] > (value - 2)
  projverts = pts[np.where(a & b)]/spacing
  projverts = projverts[:, :2]
  return projverts

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"""

fig = plt.figure()
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)

show_in_3axis(ax1, ax2, ax3, img, spacing, mesh[0], 'r')
plt.show()
