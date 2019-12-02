from main import extract_data
from model import CystCNN
from predictor import prepare_input, post_process
import numpy as np
import cv2
load_from = 31
model_file = f'models_v2/model_{load_from:04}.hd5'
# model_file = './my_model.hd5'
model = CystCNN(model_file)
info = extract_data()
dice_av = 0
count = 0


def calculate_dice(gt, out):
    size_out = np.count_nonzero(out)
    size_gt = np.count_nonzero(gt)
    size_common = np.count_nonzero(out * gt)
    return 2 * size_common / (size_gt + size_out)


for i in range(len(info)):
    for j in range(len(info[i][1:len(info[i])]) - 1):
        if 'gt' in info[i][j]:
            model_input = prepare_input(i+1, j)
            model_output = model.predict(model_input['X'])
            out, _ = post_process(model_input, model_output)
            dice = calculate_dice(info[i][j]['gt'], out)
            print(f'dice coefficient for ({i+1:02}, {j:02}):', dice)
            dice_av += dice
            count += 1
dice_av = dice_av / count
print('AVERAGE DICE COEFFICIENT:', dice_av)
