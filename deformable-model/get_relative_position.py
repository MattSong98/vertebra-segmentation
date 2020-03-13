"""Get relative positions for detection: subject 1 only
"""
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from file_importer import read_nii


class Interactor:
    """Note: self.xyz denotes volumetric position rather than physical position
        图像的xyz怎么跟data相匹配呢？
    """
    def __init__(self, img, spacing, axis):
        self.click = 0
        self.x = 0
        self.y = 0
        self.z = 0
        self.img = img
        self.spacing = spacing
        self.axis = axis

        self.axis.imshow(img[int(img.shape[0] / 2), :, :], cmap='gray', aspect=spacing[1]/spacing[2])
        self.position = list()

    def on_click(self, event):

        if self.click == 0:
            self.click += 1
            self.y = int(event.ydata)
            self.z = int(event.xdata)
            self.axis.imshow(self.img[:, self.y, :], cmap='gray', aspect=spacing[0]/spacing[2])
            plt.show()

        elif self.click == 1:
            self.click += 1
            self.x = int(event.ydata)
            self.position = [self.x, self.y, self.z]
            print("done!")

"""Main
"""
print('Reading subject1_original...')
original_fname = '/Users/haoyuesong/PycharmProjects/DeformationModel/Database/data1_prediction/subject1/data_original.nii.gz'
img, spacing = read_nii(original_fname, normalize=True)
positions = list()

for vertebra in range(1,6):
    print("click L{} to compute the relative position...".format(vertebra))
    plt.ion()
    plt.show()
    fig, axis = plt.subplots()
    inter = Interactor(img, spacing, axis)
    fig.canvas.mpl_connect('button_press_event', inter.on_click)
    plt.pause(6)
    position = [a*b for a, b in zip(inter.position, spacing)]
    positions.append(position)

print("Vertebrae's positions:", positions)
np.save('/Users/haoyuesong/PycharmProjects/DeformationModel/Database/data0_others/vertebrae_positions.npy', positions)



