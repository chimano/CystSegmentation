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
from sklearn.cluster import KMeans

def revert_crop(img, center, dest=(256, 512)):
    reverted = np.zeros((dest))
    reverted[:, center - img.shape[1]//2: center + img.shape[1]//2] = img
    return reverted


def prepare_input(n_subject, n_slice):
    result = {}
    mat = scipy.io.loadmat(f'2015_BOE_Chiu/Subject_{n_subject:02}.mat')
    imgs = mat['images'][:, :, n_slice - 1: n_slice + 2]
    scaled_imgs = [scale_image(imgs[:, :, i]) for i in range(imgs.shape[2])]
    # center = find_image_center(scaled_imgs[1])
    # imgs_roi = [[extract_roi(img[:,i * 250:i*250 + 250], center) for img in scaled_imgs] for i in range(2)]
    denoised_imgs = [[normalize_img(denoise_image(img[:,i * 125:i*125 + 250])) for img in scaled_imgs] for i in range(3)]

    result['X'] = prepare_X(denoised_imgs)
    result['scaled'] = scaled_imgs[1]
    return result


def apply_k_means(img, bin_map):
    masked = img * bin_map
    mask_1d = masked.reshape((masked.shape[0] * masked.shape[1],1))
    cluster = KMeans(n_clusters=3).fit(mask_1d)
    median_index = np.where(cluster.cluster_centers_ == np.median(cluster.cluster_centers_))[0][0]
    for i in range(len(bin_map)):
        for j in range(len(bin_map[i])):
            bin_map[i][j] = 0 if cluster.predict([[img[i][j]]])[0] != median_index else bin_map[i][j]
    
    return bin_map


if __name__ == "__main__":
    model = CystCNN('./my_model.hd5')
    model_input = prepare_input(2, 50)
    model_output = model.predict(model_input['X'])

    model_output[model_output >= 0.2] = 1.0
    model_output[model_output < 0.2] = 0.0

    center_output = model_output[1]
    model_output = np.concatenate((model_output[0],model_output[2]), 1)

    center_output = cv2.resize(center_output, (250, 256))
    model_output = cv2.resize(model_output, (500, 256))
    model_output[:,125:375] = center_output
    img = model_input['scaled']

    model_output[model_output >= 1] = 1.0
    model_output = apply_k_means(img, model_output)

    model_output = np.array(model_output, dtype=np.uint8) * 255
    img = np.stack((model_input['scaled'],)*3, axis=-1)
    # img = cv2.imread('./gt_output/Subject_05_23.png')
    model_output = cv2.applyColorMap(model_output, cv2.COLORMAP_JET)
    fin = cv2.addWeighted(model_output, 0.6, img, 0.4, 0)
    plt.imshow(fin)
    plt.show()
