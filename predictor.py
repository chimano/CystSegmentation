import json
from os import listdir
from os.path import isdir, isfile, join

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.io

from model import CystCNN, prepare_X
from preprocessor import (denoise_image, extract_roi, find_image_center,
                          normalize_img, scale_image, get_mask)
from sklearn.cluster import KMeans


def revert_crop(img, center, dest=(256, 512)):
    reverted = np.zeros((dest))
    reverted[:, center - img.shape[1]//2: center + img.shape[1]//2] = img
    return reverted


def prepare_input(n_subject, n_slice):
    result = {}
    mat = scipy.io.loadmat(f'2015_BOE_Chiu/Subject_{n_subject:02}.mat')
    imgs = mat['images'][:, :, n_slice - 1: n_slice + 2]
    scaled_imgs = [scale_image(
        imgs[:, :, i]) * get_mask(n_subject, n_slice) for i in range(imgs.shape[2])]
    # center = find_image_center(scaled_imgs[1])
    # imgs_roi = [[extract_roi(img[:,i * 250:i*250 + 250], center) for img in scaled_imgs] for i in range(2)]
    denoised_imgs = [[normalize_img(denoise_image(
        img[:, i * 125:i*125 + 250])) for img in scaled_imgs] for i in range(3)]

    result['X'] = prepare_X(denoised_imgs)
    result['scaled'] = scale_image(imgs[:, :, 1])
    result['subject'] = n_subject
    result['n_slice'] = n_slice
    return result


def apply_k_means(img, bin_map):
    masked = img * bin_map
    mask_1d = masked.reshape((masked.shape[0] * masked.shape[1], 1))
    cluster = KMeans(n_clusters=3).fit(mask_1d)
    max_index = np.where(cluster.cluster_centers_ ==
                         max(cluster.cluster_centers_))[0][0]
    kmeans_result = cluster.labels_.reshape((256, 500))
    for i in range(kmeans_result.shape[0]):
        for j in range(kmeans_result.shape[1]):
            kmeans_result[i, j] = 1 if kmeans_result[i, j] != max_index else 0
    return kmeans_result * bin_map


def post_process(model_input, model_output, b_apply_k_means=True):
    center_output = model_output[1]
    model_output = np.concatenate((model_output[0], model_output[2]), 1)

    center_output = cv2.resize(center_output, (250, 256))
    model_output = cv2.resize(model_output, (500, 256))
    model_output[:, 125:375] = center_output
    model_output = (model_output - model_output.min()) / \
        (model_output.max() - model_output.min())
    img = model_input['scaled']
    mask = get_mask(model_input['subject'], model_input['n_slice'])

    model_output[model_output >= 0.8] = 1.0
    model_output[model_output < 0.8] = 0.0
    model_output = model_output * mask

    if b_apply_k_means:
        model_output = apply_k_means(denoise_image(img), model_output)

    model_output = np.array(model_output * 255, dtype=np.uint8)
    display_model_output = np.array(model_output, dtype=np.uint8)
    img = np.stack((model_input['scaled'],)*3, axis=-1)
    display_model_output = cv2.applyColorMap(
        display_model_output, cv2.COLORMAP_JET)
    fin = cv2.addWeighted(display_model_output, 0.6, img, 0.4, 0)
    return (model_output, fin)


if __name__ == "__main__":
    model = CystCNN('my_model.hd5')
    subject_number = 1
    slice_number = 32
    model_input = prepare_input(subject_number, slice_number)
    model_output = model.predict(model_input['X'])
    _, slice_with_output = post_process(model_input, model_output)
    images_with_titles = [
        (slice_with_output, 'Retina with predicted cyst regions')]
    try:
        actual_cyst = cv2.imread(
            f'gt_output/Subject_{subject_number:02}_{slice_number:02}.png', cv2.IMREAD_GRAYSCALE)
        display_gt = cv2.applyColorMap(
            actual_cyst, cv2.COLORMAP_JET)
        img = np.stack((model_input['scaled'],)*3, axis=-1)
        slice_with_gt = cv2.addWeighted(
            display_gt, 0.6, img, 0.4, 0)
        images_with_titles.append(
            (slice_with_gt, 'Retina with ground truth'))
    except:
        print(
            f'Could not find corresponding ground truth.\nSubject:{subject_number:02}\nSlice:{slice_number:02}')

    fig = plt.figure()

    for n, (image, title) in enumerate(images_with_titles):
        a = fig.add_subplot(1,
                            len(images_with_titles), n + 1)
        plt.imshow(image)
        a.set_title(title)
    plt.show()
