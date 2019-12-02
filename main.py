import json
import os

import cv2
import numpy as np
import scipy.io

from preprocessor import (denoise_image, extract_roi, find_image_center,
                          normalize_img, scale_image, prepare_cakes, get_mask)


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
        if np.count_nonzero(info.get('gt', 0)):
            cv2.imwrite(
                f'gt_output/Subject_{i:02}_{j:02}.png', info['gt']*255)
        cv2.imwrite(
            f'output/Subject_{i:02}_{j:02}.png', info['scaled_image'])
        cv2.imwrite(
            f'output/Subject_{i:02}_{j:02}_denoised.png', info['denoised'])
    except:
        pass


def extract_data(path='./2015_BOE_Chiu', b_write_to_disk=False):
    info = []
    for i in range(1, 11):
        subject_info = []
        mat = scipy.io.loadmat(f'2015_BOE_Chiu/Subject_{i:02}.mat')
        images = mat['images']
        grader1 = mat['manualFluid1']
        grader2 = mat['manualFluid1']
        for j in range(len(images[0][0])):
            info_dict = {}
            mf1 = np.nan_to_num(scale_image(
                grader1[:, :, j]), nan=0.0)
            mf2 = np.nan_to_num(scale_image(
                grader2[:, :, j]), nan=0.0)

            newimg = scale_image(images[:, :, j])
            try:

                denoised = denoise_image(newimg) * get_mask(i, j)
                if j == 32:
                    a = 1
                info_dict['i'] = i
                info_dict['j'] = j
                info_dict['img'] = images[:, :, j]
                info_dict['scaled_image'] = newimg
                info_dict['denoised_normalized'] = normalize_img(denoised)
                info_dict['denoised'] = denoised
                if np.count_nonzero(mf1) and np.count_nonzero(mf2):
                    ground_truth = generate_ground_truth(mf1, mf2)
                    info_dict['gt'] = ground_truth
                if b_write_to_disk:
                    output_to_disk(info_dict)
            except:
                pass
            subject_info.append(info_dict)
        info.append(subject_info)
        print(f'Parsed subject {i:02}')
    return info


def prepare_training_data(info):
    for i in range(len(info)):
        for j in range(len(info[i][1:len(info[i])]) - 1):
            if 'gt' in info[i][j]:
                for l in range(3):
                    layer_mask = get_mask(i+1, j)[:, l * 125:l*125 + 250]
                    try:
                        cake_stack = np.stack([prepare_cakes(info[i][k]['denoised_normalized'][:, l * 125:l*125 + 250] * layer_mask)
                                               for k in range(j-1, j+2)], axis=2)
                        cake = prepare_cakes(
                            info[i][j]['denoised_normalized'][:, l * 125:l*125 + 250] * layer_mask)
                        train_info = {
                            'x': [cake_stack.tolist(), cake.tolist()],
                            'y': cv2.resize(info[i][j]['gt'][:, l * 125:l*125 + 250], (125, 128)).tolist()
                        }
                        with open(f'./train/s_{i+1:02}-{j:02}-{l}', 'w') as f:
                            json.dump(train_info, f)
                    except:
                        continue


if __name__ == "__main__":
    info = extract_data(b_write_to_disk=False)
    prepare_training_data(info)
