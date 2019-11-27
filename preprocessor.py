
import math

import cv2
import numpy as np
from abel.tools.center import fit_gaussian
from skimage.restoration import denoise_tv_chambolle
from skimage.transform import radon


def extract_roi(img, xc, width=250):
    if width//2 + xc > img.shape[1]:
        xc = img.shape[1] - width//2
    if xc - width//2 < 0:
        xc = width//2

    return img[:, xc - width//2: xc + width//2]


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


def normalize_img(img):
    img = img - np.average(img)
    img = img/np.var(img)
    return img


def find_image_center(img):
    projection = project_into_columns(img)
    return int(fit_gaussian(projection)[1])


def denoise_image(img):
    return denoise_tv_chambolle(img, weight=0.08) * 255


def scale_image(img):
    return cv2.resize(img, (500, 256))
