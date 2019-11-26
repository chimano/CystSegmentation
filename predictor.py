import json
from os import listdir
from os.path import isdir, isfile, join

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.io

from model import CystCNN, prepare_X
from preprocessor import (denoise_image, extract_roi, find_image_center,
                          normalize_img, scale_image)


def revert_crop(img, center, dest=(256, 512)):
    reverted = np.zeros((dest))
    reverted[:, center - img.shape[1]//2: center + img.shape[1]//2] = img
    return reverted


def prepare_input(n_subject, n_slice):
    result = {}
    mat = scipy.io.loadmat(f'2015_BOE_Chiu/Subject_{n_subject:02}.mat')
    imgs = mat['images'][:, :, n_slice - 1: n_slice + 2]
    scaled_imgs = [scale_image(imgs[:, :, i]) for i in range(imgs.shape[2])]
    center = find_image_center(scaled_imgs[1])
    imgs_roi = [extract_roi(img, center) for img in scaled_imgs]
    denoised_imgs = [normalize_img(denoise_image(img)) for img in imgs_roi]

    result['X'] = prepare_X(denoised_imgs)
    result['center'] = center
    result['scaled'] = scaled_imgs[1]
    return result


if __name__ == "__main__":
    model = CystCNN('./my_model.hd5')
    model_input = prepare_input(1, 35)
    model_output = model.predict(model_input['X'])[0]

    model_output[model_output >= 0.12] = 1.0
    model_output[model_output < 0.12] = 0.0
    model_output = cv2.resize(model_output, (250, 256)) * 255
    model_output = revert_crop(model_output, model_input['center'])
    model_output = np.array(model_output, dtype=np.uint8)
    img = np.stack((model_input['scaled'],)*3, axis=-1)
    model_output = cv2.applyColorMap(model_output, cv2.COLORMAP_JET)
    fin = cv2.addWeighted(model_output, 0.6, img, 0.4, 0)
    plt.imshow(fin)
    plt.show()
