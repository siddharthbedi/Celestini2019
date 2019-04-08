#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 20:05:57 2019

@author: siddharth
"""

import numpy as np
import pandas as pd

data=pd.read_csv('zoo_data.csv',header=None)

data.columns=['NAME','HAIR','FEATHERS','EGGS','MILK','AIRBORNE','AQUATIC','PREDATOR','TOOTHED','BACKBONE','BREATHERS','VENOM','FINS','LEGS','TAIL','DOMESTIC','CATSIZE','TYPE']

data.head()

x=data.iloc[:,1:17]
y=data.iloc[:,17:18]



from keras.models import Sequential
from keras.layers import Activation,Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
import keras

from sklearn import preprocessing
min_max_scalar=preprocessing.MinMaxScaler()
y_scale=min_max_scalar.fit_transform(y)
x_scale=min_max_scalar.fit_transform(x)

from sklearn.model_selection import train_test_split as tts
x_train,x_val,y_train,y_val = tts(x_scale,y_scale,test_size=0.15,random_state=0)

# from keras.utils.np_utils import to_categorical
# y_train = to_categorical(y_train)
# y_val = to_categorical(y_val)

y_train.shape

model = Sequential()
model.add(Dense(200, activation = "relu",input_shape=(16,)))
model.add(BatchNormalization())
model.add(Dense(120, activation = "relu"))
# model.add(BatchNormalization())

# model.add(Dense(100, activation = "relu"))
# # model.add(BatchNormalization())

model.add(Dense(50, activation = "relu"))
# # model.add(BatchNormalization())

# model.add(Dense(25, activation = "relu"))
model.add(Dense(1, activation = "sigmoid"))

print(model.summary())

from keras import optimizers
new = optimizers.Adam(lr=0.001)

from keras import backend as K
def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

model.compile(loss='binary_crossentropy', optimizer =new, metrics=['accuracy',recall_m,precision_m])

trained_model = model.fit(x=x_train, y=y_train, epochs=20,validation_data=(x_val, y_val))

import matplotlib.pyplot as plt
plt.plot(trained_model.history['acc'])
# plt.plot(trained_model.history['val_loss'])
plt.title('model loss')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
# plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(trained_model.history['recall_m'])
plt.plot(trained_model.history['precision_m'])
plt.title('recall-precision')
plt.ylabel('value')
plt.xlabel('Epochs')
# plt.legend(['train', 'test'], loc='upper left')
plt.show()