#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 12:58:44 2019

@author: siddharth
"""

import numpy as np
import pandas as pd

data=pd.read_csv('zoo_data.csv',header=None)

data.columns=['NAME','HAIR','FEATHERS','EGGS','MILK','AIRBORNE','AQUATIC','PREDATOR','TOOTHED','BACKBONE','BREATHERS','VENOM','FINS','LEGS','TAIL','DOMESTIC','CATSIZE','TYPE']

data.head()

import matplotlib.pyplot as plt

plt.matshow(data.corr())
plt.show()

corr = data.corr()
corr.style.background_gradient(cmap='coolwarm').set_precision(2)

data = data.drop(['HAIR','EGGS'],axis=1)

data.head()

x=data.iloc[:,1:15]
y=data['TYPE']

x.head()

y.head()

y.shape

legs = pd.get_dummies(x['LEGS'])
x = pd.concat([x.drop(['LEGS'],axis=1),legs],axis=1)
y = pd.get_dummies(y)

y.head()

y.shape

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

from keras.models import Sequential
from keras.layers import Activation,Dense, Dropout
from keras import backend as K
import keras

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

model = Sequential()
model.add(Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001),input_dim=19))
model.add(Dense(32, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)))
model.add(Dense(16, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)))
model.add(Dense(7, activation='softmax'))

print(model.summary())

model.compile(optimizer='Adam',
              loss='categorical_crossentropy',
              metrics=['accuracy',recall_m,precision_m])

trained_model = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=(0.2))

import matplotlib.pyplot as plt
plt.plot(trained_model.history['acc'])
plt.title('model loss')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.show()

plt.plot(trained_model.history['loss'])
plt.plot(trained_model.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('Epochs')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(trained_model.history['recall_m'])
plt.plot(trained_model.history['precision_m'])
plt.title('recall-precision')
plt.ylabel('value')
plt.xlabel('Epochs')
plt.legend(['recall', 'precision'], loc='upper left')
plt.show()
