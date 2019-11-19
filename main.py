import os

import cv2
import matplotlib.pyplot as plt
import mpmath
import numpy as np
import scipy.io
from abel.tools.center import find_center
from scipy.optimize import curve_fit
from tvd import TotalVariationDenoising
from skimage.transform import radon


def project_into_columns(img):
    # Need to understand how the paper did this part
    proj = radon(img, theta=[90], circle=True)[:, 0]
    return proj


def extract_roi(img, xc, width=250):
    if width//2 + xc > img.shape[1]:
        xc = img.shape[1] - width//2
    if xc - width//2 < 0:
        xc = width//2

    return img[:, xc - width//2: xc + width//2]


def gaus(x, a, x0, sigma):
    return a*exp(-(x-x0)**2/(2*sigma**2))


if not os.path.isdir('./output'):
    os.mkdir('output')
for i in range(1, 11):
    mat = scipy.io.loadmat(f'2015_BOE_Chiu/Subject_{i:02}.mat')
    images = mat['images']
    manualfluid = mat['manualFluid1']
    for j in range(len(images[0][0])):
        newimg = cv2.resize(images[:, :, j], (512, 256))
        newfluid = cv2.resize(manualfluid[:, :, j], (512, 256))
        projection = project_into_columns(newimg)
        try:

            a = int(fit_gaussian(projection)[1])
            roi = extract_roi(newimg, a)
            denoised = TotalVariationDenoising(roi).generate()
            cv2.imwrite(f'output/Subject_{i:02}_{j:02}.png', newimg)
            cv2.imwrite(f'output/Subject_{i:02}_{j:02}_denoised.png', roi)
            cv2.imwrite(f'output/Subject_{i:02}_{j:02}_denoised.png', denoised)
        except:
            pass
