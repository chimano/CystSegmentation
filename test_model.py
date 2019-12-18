from main import extract_data
from model import CystCNN
from predictor import prepare_input, post_process
import numpy as np
import cv2
# load_from = 31
# model_file = f'models_v2/model_{load_from:04}.hd5'
model_file = './my_model.hd5'
model = CystCNN(model_file)
info = extract_data()
dice_av = 0
precision_av = 0
recall_av = 0
count = 0


def calculate_dice(gt, out):
    size_out = np.count_nonzero(out)
    size_gt = np.count_nonzero(gt)
    size_common = np.count_nonzero(out * gt)
    return 2 * size_common / (size_gt + size_out)


def calculate_precision(gt, out):
    true_positive = gt*out
    delta = out - gt
    delta[delta <= 0] = 0
    false_positive = delta
    tp_count = np.count_nonzero(true_positive)
    fp_count = np.count_nonzero(false_positive)
    return tp_count/(tp_count + fp_count)


def calculate_recall(gt, out):
    true_positive = gt*out
    delta = gt - out
    delta[delta <= 0] = 0
    false_negative = delta
    tp_count = np.count_nonzero(true_positive)
    fn_count = np.count_nonzero(false_negative)
    return tp_count/(tp_count + fn_count)


for i in range(len(info)):
    for j in range(len(info[i][1:len(info[i])]) - 1):
        if 'gt' in info[i][j]:
            model_input = prepare_input(i+1, j)
            model_output = model.predict(model_input['X'])
            out, _ = post_process(model_input, model_output)
            dice = calculate_dice(info[i][j]['gt'], out)
            recall = calculate_recall(info[i][j]['gt'], out)
            precision = calculate_precision(info[i][j]['gt'], out)
            print(f'dice coefficient for ({i+1:02}, {j:02}):', dice)
            print(f'recall for ({i+1:02}, {j:02}):', recall)
            print(f'precision for ({i+1:02}, {j:02}):', precision)
            print('-'*50)
            dice_av += dice
            recall_av += recall
            precision_av += precision
            count += 1
dice_av = dice_av / count
print('AVERAGE DICE COEFFICIENT:', dice_av)
precision_av = precision_av / count
print('AVERAGE PRECISION:', precision_av)
recall_av = recall_av / count
print('AVERAGE RECALL:', recall_av)
