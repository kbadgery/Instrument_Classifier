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
from sklearn.metrics import recall_score, confusion_matrix, accuracy_score

from music_functions import *

X = np.load("Processed Data/50000_multilabel_samples_X.npy")
y = np.load("Processed Data/50000_multilabel_samples_y.npy")



X_train, X_test, y_train, y_test = train_test_split(X, y[:,0], test_size = 0.1, random_state = 2)

input_shape = (len(X[0]), 1)
n_filters =50
kernel_size = 1000


# Model Architecture

classifier = Sequential()
classifier.add(Conv1D(n_filters, kernel_size, input_shape = input_shape, strides=1, padding='same',activation = 'relu'))
#classifier.add(MaxPooling1D(pool_size=10, strides=None))
classifier.add(GlobalMaxPooling1D())
#classifier.add(Flatten())
classifier.add(Dense(units = 15, activation = 'relu'))
#classifier.add(Dropout(0.2))
classifier.add(Dense(units = 15, activation = 'relu'))
#classifier.add(Dense(units = 10, activation = 'relu'))
#classifier.add(Dropout(0.2))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling

opt = Adam(learning_rate=0.0001)
classifier.compile(optimizer = opt, loss = 'binary_crossentropy', metrics = ['accuracy','mean_squared_error', Recall()])

# Training

classifier.fit(X_train, y_train, batch_size = 100, epochs = 5)

classifier2 = classifier

# Validation

# Print metrics for training set
y_train_pred = np.round(classifier.predict(X_train))
recall = recall_score(y_train, y_train_pred, average='binary')
print("Training Recall:", recall)

# Print metrics for validation set
y_pred = np.round(classifier.predict(X_test))
recall = recall_score(y_test, y_pred, average='binary')
accuracy = accuracy_score(y_test, y_pred)
print("Validation Recall:", recall)
print("Validation Accuracy:", accuracy)

