# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 19:53:51 2020

@author: Kip
"""

import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

from music_functions import *

# output file names
X_filename_out = "single_instruments_1000_X_test_set"
y_filename_out = "single_instruments_1000_y_test_set"

# user inputs
n_freq_buckets = 1000 # number of logarithmically space frequency buckets for CNN input aka CNN input dimensionality
max_freq = 15000 # largest frequency bin
sample_rate = 16000

data_path='Test Data - Raw'

# load data
print("Loading Data:")
wavs = readWavs(data_path)

# read class labels and perform one-hot-encoding
class_labels = [i.split('_')[0] for i in list(wavs)]
encoder = LabelEncoder()
encoder.fit(class_labels)
encoded_y = encoder.transform(class_labels)
y = np_utils.to_categorical(encoded_y)

# take FFT of every wav
print("Taking DFTs:")
freq_bin_array = np.geomspace(1, max_freq, num=n_freq_buckets) # creates logarithmically spaced frequency buckets
fft = bulkFFTer(wavs.to_numpy(), freq_bin_array, sample_rate)

# Add dimension so Conv1D layer likes it
X = np.expand_dims(fft, axis = 2)

# save numpy arrays
np.save("Processed Data - Test\\" + X_filename_out, X)
np.save("Processed Data - Test\\" + y_filename_out, y)
np.save("Processed Data - Test\\Unique Class Labels - Ordered", np.asarray(list(OrderedDict.fromkeys(class_labels))))
