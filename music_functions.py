# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 15:08:16 2020

@author: Kip
"""

import matplotlib.pyplot as plt
from scipy.io import wavfile as wav
from scipy.fftpack import fft, fftfreq
import random
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from sklearn.metrics import recall_score, confusion_matrix

def findNearest(array, value): # returns the value and index of array element that is nearest to the input value
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx

def wav_fft(data, rate): # returns the frequency vector (x-axis) and amplitudes (y-axis) of fourier transform of input
    
    fft_out = fft(data)
    freq_vector = fftfreq(len(fft_out), 1.0/rate)
    fft_trim = fft_out[range(len(fft_out)//2)]
    freq_vector_trim = freq_vector[range(len(fft_out)//2)]
    
    return freq_vector_trim, abs(fft_trim)

def reBucket(fft_amps,fft_freqs, freq_bins): # creates "lower resolution" fourier transforms with less frequency buckets
    
    bucketed_array = np.zeros([len(freq_bins),2])
    bucketed_array[:,0] = freq_bins
    
    for i in range(0,len(fft_freqs)):        
        # find nearest freq
        nearest, index = findNearest(freq_bins,fft_freqs[i])
        # increase amplitude sum of nearest freq
        bucketed_array[index,1] = bucketed_array[index,1] + fft_amps[i]
        #bucketed_array[index,2] += 1
               
        
    bucketed_array[np.isnan(bucketed_array)] = 0    
    
    return bucketed_array

def reBucket_avg(fft_amps,fft_freqs, freq_bins): # creates "lower resolution" fourier transforms with less frequency buckets
    
    bucketed_array = np.zeros([len(freq_bins),3])
    bucketed_array[:,0] = freq_bins
    
    for i in range(0,len(fft_freqs)):        
        # find nearest freq
        nearest, index = findNearest(freq_bins,fft_freqs[i])
        # increase amplitude sum of nearest freq
        bucketed_array[index,1] = bucketed_array[index,1] + fft_amps[i]
        bucketed_array[index,2] += 1
               
        
    bucketed_array[np.isnan(bucketed_array)] = 0    
    bucketed_array[:,3] = bucketed_array[:,1]/bucketed_array[:,2]
    
    return bucketed_array


def readWavs(directory): # read in each file from path and create df containing wav data
    
    df = pd.DataFrame()
    i = 0
    for filename in tqdm(os.listdir(directory)):
        rate, data = wav.read(directory + "\\" + filename)
        # some normalize rate thing here
        
        df[filename] = data       
        i = i + 1    
        
        #if i % 100 == 0:
        #    print(i)
        
    return df

def bulkFFTer(input_wavs, freq_bin_array, rate): # takes dataframe containing all records and convert to rows of training data
    
    fft_out = []         
        
    for i in tqdm(range(len(input_wavs[0]))):
    
        freq_vector, fft = wav_fft(input_wavs[:,i], rate)    
        bucketed_fft = reBucket(fft, freq_vector, freq_bin_array)
          
        fft_out.append(bucketed_fft[:,1])
              
        #if i % int(0.01*len(input_wavs[0])) == 0:
        #   print(str((i//len(input_wavs[0])*100)+1) + "%")
                    
    return np.asarray(fft_out)


def removeDupes(seq): # returns a list that removes duplicates while maintaining order
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


def createCombos(wav_array, max_combinations, labels): # creates synthetic data that creates combinations of instrument wav files
    # max combinations is the max number of instruments that can be combined for each sample
    # randomly select integer between 1 and 4? - this number represents the number of instrument samples to combine
    
    n_to_combine = random.randint(1,max_combinations)
    # randomly select integer between 0 and n_samples in the wav df, 
    random_columns = random.sample(range(0,len(wav_array[0])),n_to_combine)    
    synth_wav = wav_array[:,random_columns[0]]   
    
    y_array = labels[random_columns[0],:]

    for j in range (1, len(random_columns)):                
        synth_wav =  synth_wav + wav_array[:,random_columns[j]]
        y_array = y_array + labels[random_columns[j],:]
        
        
    return synth_wav, y_array

def bulkCombos(wav_array, max_combinations, n_to_make, labels): # uses create combos function to create synthetic badat in bulk
    
    bulkSynth_wavs = []
    bulk_y = []

    for i in tqdm(range(n_to_make)):
        
        synth_wav, y_array = createCombos(wav_array, max_combinations, labels)
        bulkSynth_wavs.append(synth_wav)
        bulk_y.append(y_array)
        
    
    bulkSynth_wavs = np.asarray(bulkSynth_wavs)
    bulk_y = np.asarray(bulk_y)
            
    return np.transpose(bulkSynth_wavs), bulk_y

def calcMetrics(y_true, y_pred):
    
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0
    
    for i in range(0,len(y_true)): # loop through the rows    
        for j in range(0,len(y_true[0])): # loops through the columns    
            if y_true[i,j] == y_pred[i,j]: # if the prediction is correct            
                if y_true[i,j] == 1:
                    true_positives += 1
                else:
                    true_negatives += 1                 
            else:
                if y_true[i,j] == 1: # if the predictino is wrong, and it should have predicted a positive value
                    false_negatives += 1
                else:
                    false_positives += 1 
                    
    recall = true_positives / (true_positives+false_negatives)
    precision = true_positives / (true_positives+false_positives)
    return recall, precision
   
def IC_recall(y_pred, y_test):        
    y_pred_1d = np.argmax(y_pred, axis=1) # un-one-hot-encode data
    y_test_1d = np.argmax(y_test, axis=1) 
    return recall_score(y_test_1d, y_pred_1d, average='micro')

def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """    
    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()