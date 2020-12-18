# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 15:04:42 2020

@author: Kip
"""
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

from music_functions import *

# output file names
X_filename_out = "50000_multilabel_samples_X"
y_filename_out = "50000_multilabel_samples_y"

# user inputs
n_freq_buckets = 1000 # number of logarithmically space frequency buckets for CNN input aka CNN input dimensionality
max_instruments_to_combine = 6 # maximum number of instruments that will be combined in one synthetic sample
max_freq = 15000 # largest frequency bin
n_synth_samples = 50000 # number of synthetic samples to create

sample_rate = 16000

data_path='Data'

# load data
wavs = readWavs(data_path)

# read class labels and hot-encode them
class_labels = [i.split('_')[0] for i in list(wavs)]
encoder = LabelEncoder()
encoder.fit(class_labels)
encoded_y = encoder.transform(class_labels)
hot_encoded_y = np_utils.to_categorical(encoded_y)

# create synthetic multilabel data by combining instrument wav files
synth_wav_data, y = bulkCombos(wavs.to_numpy(), max_instruments_to_combine, n_synth_samples, hot_encoded_y)
y[y > 1] = 1

# take FFT of every wav
freq_bin_array = np.geomspace(1, max_freq, num=n_freq_buckets) # creates logarithmically spaced frequency buckets
fft_synth = bulkFFTer(synth_wav_data, freq_bin_array, sample_rate)

# Add dimension so Conv1D layer likes it
X = np.expand_dims(fft_synth, axis = 2)

# Save data
np.save(X_filename_out, X)
np.save(y_filename_out, y)