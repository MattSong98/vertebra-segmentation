from mesh_util import vvLUT
from file_importer import read_ply, read_nii

v, f = read_ply("/Users/haoyuesong/PycharmProjects/DeformationModel/Database/data4_simplified/L5.ply")
print(f)
LUT = vvLUT(v, f)
print(LUT)