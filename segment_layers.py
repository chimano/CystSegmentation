import numpy as np
import scipy.io
import cv2
import math

def segment_layers(img):
    padded_immage = pad_image(img)
    gradient_image, gradient_image_minus = vertical_gradient_inverse(padded_immage)

    return 

def pad_image(img): 
    new_image = cv2.copyMakeBorder(img, 0, 0, 1, 1, cv2.BORDER_CONSTANT)
    return new_image

def vertical_gradient_inverse(img):
    image_size = np.shape(img)
    gradient_image = np.full(image_size, np.nan)
    new_img = convert_top_rows(img)
    for i in range (0, image_size(1)):
        gradient_image[:,i] = -1*np.gradient(img[:,i],2)
    return gradient_image, (gradient_image*-1 + 1)

def ind2sub(array_shape, ind):
    ind[ind < 0] = -1
    ind[ind >= array_shape[0]*array_shape[1]] = -1
    rows = (ind.astype('int') / array_shape[1])
    cols = ind % array_shape[1]
    return (rows, cols)

def sub2ind(array_shape, rows, cols):
    ind = rows*array_shape[1] + cols
    ind[ind < 0] = -1
    ind[ind >= array_shape[0]*array_shape[1]] = -1
    return ind

def convert_top_rows(img):
    shape = np.shape(img)
    new_image = img
    width = shape[1]
    for i in range (0, shape[0]):
        average = sum(img[i,:])/width
        if (average == 255):
            new_image[i,:] = np.zeros(width)
    return new_image

def simple_segmentation(img): 
    img_shape = np.shape(img)
    mask = np.zeros(img_shape)
    top_indices = np.zeros(img_shape[1])
    bottom_indices = np.zeros(img_shape[1])
    new_image = convert_top_rows(img)
    for j in range(0, img_shape[1]):
        col = new_image[:,j]
        top_indices[j] = int(from_top(col))
        bottom_indices[j] = int(from_bottom(col))
        # print("col: ", j, ", bottom_index: ", bottom_indices[j], ", top_index: ", top_indices[j])
        
    new_top_indices, new_bottom_indices = smooth(top_indices, bottom_indices)
    for j in range(0, img_shape[1]):
        mask[int(new_top_indices[j]):int(new_bottom_indices[j]), j] = 1
    
    return mask

def smooth(top_indices, bottom_indices):
    length = np.shape(top_indices)[0]
    new_top_indices = np.zeros(length, dtype=np.uint8)
    new_bottom_indices = np.zeros(length, dtype=np.uint8)
    
    top_average = sum(top_indices[:])/length
    bottom_average = sum(bottom_indices[:])/length

    for i in range (0, length):
        if abs(top_indices[i] - top_average) > 30:
            top_indices[i] = top_average
        if abs(bottom_indices[i] - bottom_average) > 40:
            bottom_indices[i] = bottom_average

    new_top_indices[0] = top_indices[0]
    new_top_indices[1] = top_indices[1]
    new_top_indices[length-2] = top_indices[length-2]
    new_top_indices[length-1] = top_indices[length-1]

    new_bottom_indices[0] = bottom_indices[0]
    new_bottom_indices[1] = bottom_indices[0]
    new_bottom_indices[length-2] = bottom_indices[length-2]
    new_bottom_indices[length-1] = bottom_indices[length-1]

    count = 0
    while count < 5:
        for i in range (2, length - 2):
            start = i - 2
            end = i + 2
            if (count == 0):
                new_top_indices[i] = int(sum(top_indices[start:end])/4)
            new_bottom_indices[i] = int(sum(bottom_indices[start:end])/4)
            print ("top: ", new_top_indices[i], ", bottom: ",    new_bottom_indices[i])
        bottom_indices = new_bottom_indices
        count+= 1
    return top_indices, bottom_indices

def from_top(column):
    length = np.shape(column)[0]
    index = 0
    current_average = np.sum(column)/length
    prev_average = current_average - 5
    first_catch = 0
    count = 0
    while (count < 4 and index <= length):
        index+= 6
        length-= 6
        prev_average = current_average
        current_average = np.sum(column[index:])/length
        difference = current_average - prev_average
        if (current_average < prev_average):
            if (count == 0):
                first_catch = index
            count+=1
        else:
            count = 0
    
    index = first_catch - 6
    prev_average = 0
    current_average = 0
    length+= 6 * count 
    count = 0
    while (count < 3 and index <= length):
        index+= 1
        length-= 1
        prev_average = current_average
        current_average = np.sum(column[index:])/length
        difference = current_average - prev_average
        if (current_average < prev_average):
            if (count == 0):
                first_catch = index
            count+=1
        else:
            count = 0

    return first_catch

def from_bottom(column):
    length = np.shape(column)[0]
    index = length-1
    current_average = np.sum(column)/length
    prev_average = current_average - 15
    first_catch = 0
    count = 0
    while (count < 2 and index >= 0):
        index-= 5
        length-= 5
        prev_average = current_average
        current_average = np.sum(column[:index])/length
        difference = current_average - prev_average
        if (current_average < prev_average):
            if (count == 0):
                first_catch = index
            count+=1
        else:
            count = 0

    index = first_catch + 5
    prev_average = 0
    current_average = 0
    length+= 5 * count 
    count = 0
    while (count < 3 and index >= 0):
        index-= 1
        length-= 1
        prev_average = current_average
        current_average = np.sum(column[:index])/length
        difference = current_average - prev_average
        if (current_average < prev_average):
            if (count == 0):
                first_catch = index
            count+=1
        else:
            count = 0
    
    first_catch
    return index
