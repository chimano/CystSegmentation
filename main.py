import os

import cv2
import matplotlib.pyplot as plt
import mpmath
import numpy as np
import scipy.io
from abel.tools.center import find_center
from abel.tools.math import fit_gaussian
from scipy import exp
from scipy.optimize import curve_fit


def project_into_columns(img):
    # Need to understand how the paper did this part
    inverted = 255 - img
    proj = np.sum(inverted, 1)
    return proj


def gaus(x, a, x0, sigma):
    return a*exp(-(x-x0)**2/(2*sigma**2))


for i in range(1, 10):
    mat = scipy.io.loadmat(f'2015_BOE_Chiu/Subject_0{i}.mat')
    images = mat['images']
    manualfluid = mat['manualFluid1']
    for j in range(len(images[0][0])):
        newimg = cv2.resize(images[:, :, j], (512, 256))
        newfluid = cv2.resize(manualfluid[:, :, j], (512, 256))
        projection = project_into_columns(newimg)
        xc = fit_gaussian(projection)[1]
        cv2.imwrite(f'output/Subject_0{i}_{j}.png', newimg)
