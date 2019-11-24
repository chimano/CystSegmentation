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
import json
import math


def prepare_cakes(img, N=5, K=8):
    rows, cols = img.shape
    cakes = np.zeros((rows, cols, K + 1))
    for i in np.arange(0, 180, 180/K):
        slices = np.zeros((rows, cols, 2*N+1))
        for j in range(-N, N+1):
            translation = np.float32(
                [[1, 0, j*math.cos(math.radians(i))], [0, 1, j*math.sin(math.radians(i))]])
            translated = cv2.warpAffine(img.astype(
                np.float32), translation, (cols, rows))
            slices[:, :, j+N] = translated
        cake = np.zeros((rows, cols))
        for j in range(len(cake)):
            for k in range(len(cake[j])):
                cake[j, k] = min(slices[j, k, :])
        index = i / 180 * K
        cakes[:, :, int(index)] = cake
    cakes[:, :, K] = img
    return cakes


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
    ground_truth = mf1 * mf2
    return ground_truth


if not os.path.isdir('./output'):
    os.mkdir('output')
if not os.path.isdir('./gt_output'):
    os.mkdir('gt_output')
if not os.path.isdir('./train'):
    os.mkdir('train')


def output_to_disk(info):
    i = info['i']
    j = info['j']
    try:
        if np.count_nonzero(info.get('gt_roi', 0)):
            cv2.imwrite(
                f'gt_output/Subject_{i:02}_{j:02}.png', info['gt_roi'].tolist())
        cv2.imwrite(
            f'output/Subject_{i:02}_{j:02}.png', info['scaled_image'].tolist())
        cv2.imwrite(
            f'output/Subject_{i:02}_{j:02}_roi.png', info['roi'].tolist())
        cv2.imwrite(
            f'output/Subject_{i:02}_{j:02}_denoised.png', info['denoised'].tolist())
    except:
        pass


def extract_data(path='./2015_BOE_Chiu', b_write_to_disk=True):
    info = []
    for i in range(1, 11):
        subject_info = []
        mat = scipy.io.loadmat(f'2015_BOE_Chiu/Subject_{i:02}.mat')
        images = mat['images']
        grader1 = mat['manualFluid1']
        grader2 = mat['manualFluid1']
        for j in range(len(images[0][0])):
            info_dict = {}
            mf1 = np.nan_to_num(cv2.resize(
                grader1[:, :, j], (512, 256)), nan=0.0)
            mf2 = np.nan_to_num(cv2.resize(
                grader2[:, :, j], (512, 256)), nan=0.0)

            newimg = cv2.resize(images[:, :, j], (512, 256))
            projection = project_into_columns(newimg)
            try:

                center = int(fit_gaussian(projection)[1])
                roi = extract_roi(newimg, center)
                denoised = denoise_tv_chambolle(
                    roi, weight=0.08) * 255
                info_dict['i'] = i
                info_dict['j'] = j
                info_dict['scaled_image'] = newimg
                info_dict['roi'] = roi
                info_dict['denoised'] = denoised
                if b_write_to_disk:
                    output_to_disk(info_dict)
                if np.count_nonzero(mf1) and np.count_nonzero(mf2):
                    ground_truth = generate_ground_truth(mf1, mf2) * 255
                    gt_roi = extract_roi(ground_truth, center)
                    info_dict['gt_roi'] = gt_roi
            except:
                pass
            subject_info.append(info_dict)
        info.append(subject_info)
        print(f'Parsed subject {i:02}')
    return info


def prepare_training_data(info):
    for i in range(len(info)):
        for j in range(len(info[i][1:len(info[i])]) - 1):
            if 'gt_roi' in info[i][j]:
                try:
                    cake_stack = np.stack([prepare_cakes(info[i][k]['denoised'])
                                           for k in range(j-1, j+2)], axis=2)
                    cake = prepare_cakes(info[i][j]['denoised'])
                    train_info = {
                        'x': [cake_stack.tolist(), cake.tolist()],
                        'y': cv2.resize(info[i][j]['gt_roi'], (125, 128)).tolist()
                    }
                    with open(f'./train/s_{i:02}-{j:02}', 'w') as f:
                        json.dump(train_info, f)
                except:
                    continue


if __name__ == "__main__":
    info = extract_data(b_write_to_disk=False)
    prepare_training_data(info)
