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

def prepare_graph(img, gradient_img): 
    min_weight = 1*math.exp(-5)
    img_shape = np.shape(img)
    number_elements = np.prod(img_shape)
    adjMW = np.full((number_elements, 8), np.nan)
    adjMmW = np.full((number_elements, 8), np.nan)
    adjMX = np.full((number_elements, 8), np.nan)
    adjMY = np.full((number_elements, 8), np.nan)
    neighbor_iterator = np.array([[1, 1, 1, 0, 0, -1, -1, -1],
                         [1, 0, -1, 1, -1, 1, 0, -1]])
    
    shape_adjMW = np.shape(adjMW)
    ind = 1
    indR = 0
    while (ind != np.prod(shape_adjMW)): 
        i, j = ind2sub(shape_adjMW, ind)
        iX, iY = ind2sub(img_shape, i(0))
        jX = iX + neighbor_iterator[0, j]
        jY = iY + neighbor_iterator[1, j]
        if (jX > 1 & jX < img_shape(0) & jY > 1 & jY < img_shape(1)):
            if (jY == 1 | jY == img_shape(1)):
                adjMW[i,j] = min_weight
                adjMmW[i,j] = min_weight
            else: 
                adjMW[i,j] = 2 - gradient_img[iX, iY] - gradient_img[jX, jY] + min_weight
                adjMmW[i,j] = min_weight
            
            adjMX[i, j] = sub2ind(img_shape, iX, iY)
            adjMY[i, j] = sub2ind(img_shape, jX, jY)
        ind = ind + 1

        # Add progress here?
    keep_ind = not np.isnan(adjMW[:]) and not np.isnan(adjMY[:]) and not np.isnan(adjMY[:]) and not np.isnan(adjMmW[:])
    adjMW = adjMW[keep_ind]
    adjMmW = adjMmW[keep_ind]
    adjMX = adjMX[keep_ind]
    adjMY = adjMY[keep_ind]

