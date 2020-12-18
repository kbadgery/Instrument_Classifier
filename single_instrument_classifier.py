# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 15:16:59 2020

@author: Kip

# recall 96%  using 50 filters, 50 neuron, 50 neuron, 1000 kernel size
"""
import matplotlib.pyplot as plt
from scipy.io import wavfile as wav
from scipy.fftpack import fft, fftfreq
from random import randint
import numpy as np
import pandas as pd
import os

from keras.models import Sequential
from keras.layers import Conv1D, GlobalMaxPooling1D, MaxPooling1D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from keras import regularizers
from keras.metrics import Recall
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, confusion_matrix

from music_functions import *

X = np.load("Processed Data/single_instruments_1000_X.npy")
y = np.load("Processed Data/single_instruments_1000_y.npy")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 2)

n_classes = len(y[0])
input_shape = (len(X[0]), 1)
n_filters =70
kernel_size = 1000


# Model Architecture

classifier = Sequential()
classifier.add(Conv1D(n_filters, kernel_size, input_shape = input_shape, strides=1, padding='same',activation = 'relu'))
#classifier.add(MaxPooling1D(pool_size=10, strides=None))
classifier.add(GlobalMaxPooling1D())
#classifier.add(Flatten())
classifier.add(Dense(units = 100, activation = 'relu', )) #, kernel_regularizer='l2'))
#classifier.add(Dropout(0.2))
classifier.add(Dense(units = 100, activation = 'relu')) #, kernel_regularizer='l2'))
#classifier.add(Dense(units = 10, activation = 'relu'))
#classifier.add(Dropout(0.2))
classifier.add(Dense(units = n_classes, activation = 'softmax'))

# Compiling

opt = Adam(learning_rate=0.0001)
classifier.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics = ['accuracy','mean_squared_error', Recall()])

# Training

classifier.fit(X_train, y_train, batch_size = 1000, epochs = 1)

# Validation

y_pred = classifier.predict(X_test)
y_pred_1d = np.argmax(y_pred, axis=1) # un-one-hot-encode data
y_test_1d = np.argmax(y_test, axis=1)
recall = recall_score(y_test_1d, y_pred_1d, average='micro')
print(recall)

# same for training set
y_pred = classifier.predict(X_train)
y_pred_1d = np.argmax(y_pred, axis=1)
y_test_1d = np.argmax(y_train, axis=1)
recall = recall_score(y_test_1d, y_pred_1d, average='micro')
print(recall)