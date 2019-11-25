import keras
from keras.layers import (Concatenate, Conv2D, Conv3D, Input, MaxPooling2D,
                          MaxPooling3D, Reshape)
from keras.models import Model
from keras.optimizers import SGD
from keras.callbacks import Callback
from os import listdir
from os.path import isfile, join, isdir
import numpy as np
import json
import cv2
import matplotlib.pyplot as plt

input_1 = Input(shape=(256, 250, 3, 9))
input_2 = Input(shape=(256, 250, 9))

output_1 = Conv3D(9, (1, 1, 3))(input_1)
output_2 = Reshape((256, 250, 9))(output_1)

output_3 = MaxPooling3D(pool_size=(2, 2, 1))(input_1)
output_4 = Conv3D(9, (25, 25, 3), padding='same')(output_3)
output_5 = Conv3D(9, (1, 1, 3))(output_4)
output_6 = Reshape((128, 125, 9))(output_5)

input_2_output_2 = Concatenate()([input_2, output_2])
output_7 = Conv2D(8, (10, 10), padding='same')(input_2_output_2)
output_8 = MaxPooling2D(pool_size=(2, 2))(output_7)
output_8_output_6 = Concatenate()([output_8, output_6])
output_9 = Conv2D(8, (25, 25), padding='same')(output_8_output_6)
prob_map = Conv2D(1, (1, 1), padding='same')(output_9)
output = output_2 = Reshape((128, 125))(prob_map)

optimizer = SGD(learning_rate=0.001, momentum=0.75, nesterov=True)
model = Model(inputs=[input_1, input_2], output=output)
model.compile(optimizer, loss='binary_crossentropy')
model.summary()

with open('./train/s_04-23', 'r') as f:
    json_file = json.load(f)
    X_1= np.empty((1, 256, 250, 3, 9))
    X_2 = np.empty((1, 256, 250, 9))
    X_1[0] = np.array(json_file['x'][0])
    X_2[0] = np.array(json_file['x'][1])


model.load_weights('./my_model.h5')
ans = model.predict([X_1,X_2])
b = ans[0]
b[b>=0.2] = 1.0
b[b<0.2] = 0.0
b = cv2.resize(b, (250,256)) * 255
b = np.array(b, dtype = np.uint8)
img = cv2.imread('./output/Subject_04_23_denoised.png')
b = cv2.applyColorMap(b, cv2.COLORMAP_JET)
fin = cv2.addWeighted(b, 0.6, img, 0.4, 0)
plt.imshow(fin)
plt.show()