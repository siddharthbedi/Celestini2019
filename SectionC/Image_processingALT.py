#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 20:18:09 2019
@author: siddharth
"""


import numpy as np
import cv2
from matplotlib import pyplot as plt
from tqdm import tqdm
import os
from sklearn.metrics import mean_squared_error
from math import sqrt

from PIL import Image

def stereo_match(left_img, right_img, kernel, max_offset):
    # Load in both images, assumed to be RGBA 8bit per channel images
    #left_img = Image.open(left_img).convert('L')
    left = np.asarray(left_img)
    #right_img = Image.open(right_img).convert('L')
    right = np.asarray(right_img)
    w, h = left_img.size  # assume that both images are same size

    # Depth (or disparity) map
    depth = np.zeros((w, h), np.uint8)
    depth.shape = h, w

    kernel_half = int(kernel / 2)
    offset_adjust = 255 / max_offset  # this is used to map depth map output to 0-255 range

    for y in range(kernel_half, h - kernel_half):
        print(".", end="", flush=True)  # let the user know that something is happening (slowly!)

        for x in range(kernel_half, w - kernel_half):
            best_offset = 0
            prev_ssd = 65534

            for offset in range(max_offset):
                ssd = 0
                ssd_temp = 0

                # v and u are the x,y of our local window search, used to ensure a good
                # match- going by the squared differences of two pixels alone is insufficient,
                # we want to go by the squared differences of the neighbouring pixels too
                for v in range(-kernel_half, kernel_half):
                    for u in range(-kernel_half, kernel_half):
                        # iteratively sum the sum of squared differences value for this block
                        # left[] and right[] are arrays of uint8, so converting them to int saves
                        # potential overflow, and executes a lot faster
                        ssd_temp = int(left[y+v, x+u]) - int(right[y+v, (x+u) - offset])
                        ssd += ssd_temp * ssd_temp

                # if this value is smaller than the previous ssd at this block
                # then it's theoretically a closer match. Store this value against
                # this block..
                if ssd < prev_ssd:
                    prev_ssd = ssd
                    best_offset = offset

            # set depth output for this x,y location to the best match
            depth[y, x] = best_offset * offset_adjust

    # Convert to PIL and save it
    Image.fromarray(depth).save('depth.png')
    return depth

# morphology settings
kernel = np.ones((12,12),np.uint8)

TRAIN_DIR = '/media/acez/Storage/OpenCV/Input'
TEST_DIR = '/media/acez/Storage/OpenCV/Input/testQ'

def label_image(img):
    world_label = img.split('.')[0][2]
    if world_label == "0": return [1, 0]
    elif world_label == "1": return [0, 1]

def create_training_data():
    training_data_left = []
    training_data_right = []
    for folder in tqdm(os.listdir(TRAIN_DIR)):
        for img in tqdm(os.listdir(TRAIN_DIR+'/'+folder)):
            if img == "im0.png" or "im1.png":
                label = label_image(img)
                path = os.path.join(TRAIN_DIR+'/'+folder, img)
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if label == [1,0]:
                    training_data_left.append(np.array(img))
                elif label == [0,1]:
                    training_data_right.append(np.array(img))
            else:
                continue

            #np.save("training_data_left.npy", training_data_left)
            #np.save("training_data_right.npy", training_data_right)

    return training_data_left,training_data_right

def create_test_data():
    testing_data_left = []
    testing_data_right = []
    for folder in tqdm(os.listdir(TEST_DIR)):
        for img in tqdm(os.listdir(TEST_DIR+'/'+folder)):
            if img == "im0.png" or "im1.png":
                label = label_image(img)
                path = os.path.join(TRAIN_DIR+'/'+folder, img)
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if label == [1,0]:
                    testing_data_left.append(np.array(img))
                elif label == [0,1]:
                    testing_data_right.append(np.array(img))
            else:
                continue
            #np.save("testing_data_left.npy", testing_data_left)
            #np.save("testing_data_right.npy", testing_data_right)

    return testing_data_left,testing_data_right

# train_data_left,train_data_right = create_training_data()
# len(train_data_left)

# train_data_left = np.load('/media/acez/Storage/OpenCV/training_data_left.npy')
# train_data_right = np.load('/media/acez/Storage/OpenCV/training_data_right.npy')

#test_data_left,test_data_right = create_test_data()

test_data_left = np.load('/media/acez/Storage/OpenCV/testing_data_left.npy')
test_data_right = np.load('/media/acez/Storage/OpenCV/testing_data_right.npy')


stereo = cv2.StereoBM_create(numDisparities=64,blockSize=7)
#train_data_left
disparity,threshold,morphology = [],[],[]
for i in range(len(train_data_left)):
    #disparity.append(stereo.compute(train_data_left[i] ,train_data_right[i]))
    #np.save("disparity.npy",disparity)
    disparity = np.load('/media/acez/Storage/OpenCV/disparity.npy')
    # Apply threshold
    threshold.append(cv2.threshold(disparity[i], 0.6, 1.0, cv2.THRESH_BINARY)[1])
    # Apply morphological transformation
    morphology.append(cv2.morphologyEx(threshold[i], cv2.MORPH_OPEN, kernel))

# plt.imshow(threshold[12],'gray')

OUTPUT_DIR = '/media/acez/Storage/OpenCV/Output/trainingQ'

y_actual = []
for folder in tqdm(os.listdir(OUTPUT_DIR)):
    for img in tqdm(os.listdir(OUTPUT_DIR+'/'+folder)):
        if img == "mask0nocc.png":
            path = os.path.join(OUTPUT_DIR+'/'+folder,img)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            y_actual.append(np.array(img,dtype=np.uint16))

y_actual = np.array(y_actual)
# y_actual_temp = np.array(cv2.imread('/media/acez/Storage/OpenCV/Output/trainingQ/ArtL/mask0nocc.png',0),dtype=np.uint16)
# y_predicted = stereo_match(Image.fromarray(train_data_left[0]),Image.fromarray(train_data_right[0]),6,30)
# y_predicted = stereo_match('/media/acez/Storage/OpenCV/Input/ArtL/im0.png','/media/acez/Storage/OpenCV/Input/ArtL/im1.png',6,30)

x = Image.open('/media/acez/Storage/OpenCV/Input/ArtL/im0.png')
#type(x)

y = Image.fromarray(train_data_left[1])

#type(y)
y_predicted = []
i=0
while i<len(train_data_left)-1:
    y_predicted.append(stereo_match(Image.fromarray(train_data_left[i]),Image.fromarray(train_data_right[i+1]),6,30))

rms = sqrt(mean_squared_error(y_actual_temp, y_predicted))

#
# rms = 0
# for i in range(len(y_actual)):
#     rms += np.square(np.subtract(y_actual[i],y_predicted[i])).mean()
