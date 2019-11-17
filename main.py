import os

import cv2
import matplotlib.pyplot as plt
import mpmath
import numpy as np
import scipy.io
from abel.tools.center import find_center
from scipy.optimize import curve_fit


def project_into_columns(img):
    # Need to understand how the paper did this part
    mat = np.mat(img)
    return mat*np.linalg.inv(mat.transpose()*mat)*mat


if not os.path.isdir('./output'):
    os.mkdir('output')

for i in range(1, 10):
    mat = scipy.io.loadmat(f'2015_BOE_Chiu/Subject_0{i}.mat')
    images = mat['images']
    manualfluid = mat['manualFluid1']
    for j in range(len(images[0][0])):
        newimg = cv2.resize(images[:, :, j], (512, 256))
        newfluid = cv2.resize(manualfluid[:, :, j], (512, 256))

        cv2.imwrite(f'output/Subject_0{i}_{j}.png', newimg)
