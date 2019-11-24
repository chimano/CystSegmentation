import os

import cv2
import matplotlib.pyplot as plt
import mpmath
import numpy as np
import scipy.io
from abel.tools.center import fit_gaussian
from scipy.optimize import curve_fit
from skimage.transform import radon
from skimage.restoration import denoise_tv_chambolle


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


def generate_ground_truth(mf1, mf2):
    mf1[mf1 > 0] = 1.0
    mf2[mf2 > 0] = 1.0
    ground_truth = np.zeros(mf1.shape)
    for i in range(len(ground_truth)):
        for j in range(len(ground_truth[i])):
            ground_truth[i, j] = 1.0 if mf1[i,
                                            j] == 1.0 and mf2[i, j] == 1.0 else 0.0
    return ground_truth


if not os.path.isdir('./output'):
    os.mkdir('output')
if not os.path.isdir('./gt_output'):
    os.mkdir('gt_output')

for i in range(1, 11):
    mat = scipy.io.loadmat(f'2015_BOE_Chiu/Subject_{i:02}.mat')
    images = mat['images']
    grader1 = mat['manualFluid1']
    grader2 = mat['manualFluid1']
    for j in range(len(images[0][0])):
        mf1 = np.nan_to_num(cv2.resize(
            grader1[:, :, j], (512, 256)), nan=0.0)
        mf2 = np.nan_to_num(cv2.resize(
            grader1[:, :, j], (512, 256)), nan=0.0)

        newimg = cv2.resize(images[:, :, j], (512, 256))
        projection = project_into_columns(newimg)
        try:

            center = int(fit_gaussian(projection)[1])
            roi = extract_roi(newimg, center)
            denoised = denoise_tv_chambolle(
                roi, weight=0.08) * 255
            cv2.imwrite(f'output/Subject_{i:02}_{j:02}.png', newimg)
            cv2.imwrite(f'output/Subject_{i:02}_{j:02}_roi.png', roi)
            cv2.imwrite(f'output/Subject_{i:02}_{j:02}_denoised.png', denoised)
            if np.count_nonzero(mf1) and np.count_nonzero(mf2):
                ground_truth = generate_ground_truth(mf1, mf2) * 255
                gt_roi = extract_roi(ground_truth, center)
                cv2.imwrite(
                    f'gt_output/Subject_{i:02}_{j:02}.png', gt_roi)
        except:
            pass
