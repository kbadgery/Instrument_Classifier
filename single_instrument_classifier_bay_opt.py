# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 16:48:17 2020

@author: Kip
"""

from bayes_opt import BayesianOptimization

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

def optimize(X_train, y_train, X_test, y_test): 
    

    def objective_function(n_filters, n_layers, neurons_per_layer, l2_regularizer_yn, dropout_yn, learning_rate, batch_size, n_epochs):
        
        
        #Function with unknown internals we wish to maximize.
            
        n_classes = len(y_train[0])
        input_shape = (len(X_train[0]), 1)   
        
        n_filters =int(round(n_filters))
        kernel_size = 1000
        

            
        # Model Architecture
        
        classifier = Sequential()  
        classifier.add(Conv1D(n_filters, kernel_size, input_shape = input_shape, strides=1, padding='same',activation = 'relu'))
        classifier.add(GlobalMaxPooling1D())
        
        for i in range(int(round(n_layers))): # add fully connected layers
            
            if l2_regularizer_yn > 0:            
                classifier.add(Dense(units = int(round(neurons_per_layer)), activation = 'relu', kernel_regularizer = 'l2'))
            else:
                classifier.add(Dense(units = int(round(neurons_per_layer)), activation = 'relu'))
                
            if dropout_yn > 0 and i < int(round(n_layers)-1): # add dropout but not before final layer
                classifier.add(Dropout(0.2))
                                                  
        classifier.add(Dense(units = n_classes, activation = 'softmax'))
        
        # Compiling
        
        opt = Adam(learning_rate=learning_rate)
        classifier.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics = ['accuracy','mean_squared_error', Recall()])
        
        # Training
        
        classifier.fit(X_train, y_train, batch_size = int(round(batch_size)), epochs = int(round(n_epochs)))    
    
        # Evaluating    
    
        y_pred = classifier.predict(X_test)
        recall = recall_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1), average='micro')

    
        return recall
    
        
    # Bounded region of parameter space
    pbounds = {'n_filters': (15, 150), 'n_layers': (1, 3), 'neurons_per_layer': (20,200), 'l2_regularizer_yn': (-1,0), 'dropout_yn': (-1,0), 'learning_rate': (0.001, 0.00001), 'batch_size': (1, 1000), 'n_epochs': (20,21)}
    
    
    
    optimizer = BayesianOptimization(
        f=objective_function,
        pbounds=pbounds,
        random_state=48,
        verbose = 2
    )
    
    
    optimizer.maximize(
        init_points=2,
        n_iter=10,
    )
    
    print("Final result:", optimizer.max)
    
    return optimizer, optimizer.max


X = np.load("Processed Data/single_instruments_1000_X.npy")
y = np.load("Processed Data/single_instruments_1000_y.npy")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 2)


opt, max_opt = optimize(X_train, y_train, X_test, y_test)


opt.maximize(n_iter=5)
