import keras
from keras.layers 
import (Concatenate, Conv2D, Conv3D, Input, MaxPooling2D,
                          MaxPooling3D, Reshape)
from keras.models import Model
from keras.optimizers import SGD

epoch = 1500
batch_size = 8

input_1 = Input(shape=(250, 256, 3, 9))
input_2 = Input(shape=(250, 256, 9))

output_1 = Conv3D(9, (1, 1, 3))(input_1)
output_2 = Reshape((250, 256, 9))(output_1)

output_3 = MaxPooling3D(pool_size=(2, 2, 1))(input_1)
output_4 = Conv3D(9, (25, 25, 3), padding='same')(output_3)
output_5 = Conv3D(9, (1, 1, 3))(output_4)
output_6 = Reshape((125, 128, 9))(output_5)

input_2_output_2 = Concatenate()([input_2, output_2])
output_7 = Conv2D(8, (10, 10), padding='same')(input_2_output_2)
output_8 = MaxPooling2D(pool_size=(2, 2))(output_7)
output_8_output_6 = Concatenate()([output_8, output_6])
output_9 = Conv2D(8, (25, 25))(output_8_output_6)
prob_map = Conv2D(1, (1, 1))(output_9)

optimizer = SGD(learning_rate=0.001, momentum=0.75, nesterov=True)
model = Model(inputs=[input_1, input_2], output=prob_map)
model.compile(optimizer, loss='binary_crossentropy')

# To train...
# model.fit([input1, input2], Y,
#           validation_data=([v_input_1, v_input_2], v_Y),
#           batch_size=batch_size, epoch=epoch)
