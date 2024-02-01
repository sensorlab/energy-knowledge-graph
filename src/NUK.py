import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

#!mamba install tensorflow-gpu==2.10 -y -q

#!pip3 install tensorflow

from nilmtk import DataSet
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import defaultdict
from joblib import Memory
import pickle
from pprint import pprint

#from nilmtk import DataSet
import multiprocessing as mp

from typing import Dict

#!pip install -q scikit-learn-intelex
#from sklearnex import patch_sklearn
#patch_sklearn()

from sklearn import pipeline, metrics, linear_model, model_selection, multioutput, tree, ensemble, neural_network
#!pip install xgboost
import xgboost as xgb

import random
import numpy as np
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import accuracy_score
from sklearn import datasets, linear_model
from sklearn.model_selection import cross_val_predict
from sklearn import metrics
from sklearn.model_selection import train_test_split

import numpy as np
#import graphviz
import keras
from sklearn.utils import class_weight
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation, Conv2D, MaxPooling2D, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.preprocessing.image import ImageDataGenerator
#from ann_visualizer.visualize import ann_viz
from matplotlib import pyplot as plt 
from sklearn.metrics import multilabel_confusion_matrix, classification_report, confusion_matrix
# GPU goes brrrrrrrrrrrrrrrrrrrr
from tensorflow.keras import mixed_precision
from tensorflow.keras import regularizers
mixed_precision.set_global_policy('mixed_float16')


import numpy as np
import warnings
import math

from keras.models import Model
from keras.layers import Flatten, Dense, Input, GRU, BatchNormalization, LSTM, Bidirectional, AveragePooling1D
from keras.layers import Conv1D, Conv1DTranspose, LocallyConnected1D, SeparableConv1D, ConvLSTM1D
from keras.layers import MaxPooling1D, Dropout
from keras.layers import GlobalMaxPooling1D
from keras.layers import GlobalAveragePooling1D
from keras.preprocessing import image
#from keras.utils import layer_utils                            """!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"""
#from keras.utils.data_utils import get_file                    """!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"""
from keras import backend as K
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
#from keras.applications.imagenet_utils import _obtain_input_shape
#from keras.engine.topology import get_source_inputs

    
from io import StringIO
    
    
    
    
    
    
# GREGORJEVE
##############################################################################################    
##############################################################################################
##############################################################################################
##############################################################################################    
##############################################################################################
##############################################################################################
##############################################################################################    
##############################################################################################
##############################################################################################
##############################################################################################    
##############################################################################################
##############################################################################################
##############################################################################################    
##############################################################################################
##############################################################################################
##############################################################################################    
##############################################################################################
##############################################################################################
def test_appliance_augmentator(augmentator, *args, **kwargs):
    X, y, labels = next(augmentator(processed_data, *args, **kwargs))

    f, ax = plt.subplots()
    ax.set_title('Augmented time-series')
    ax.plot(X)
    f.show()
    
    
    
    n_appliances = kwargs.get('n_appliances_per_sample', None)
    f, ax = plt.subplots()

    
    print('Number of positive datapoints per appliance')
    for mask, label in zip(y, labels):
        if np.sum(mask) == 0:
            continue
        
        print(f'{label}:\t {np.sum(mask)}')
        

        tmp = X.copy()
        tmp[~mask] = 0
        ax.plot(tmp, linestyle=None, marker='.', label=label, alpha=0.5)
        #ax.plot(np.ma.masked_where(mask, np.full_like(X, idx)), marker='.', label=label)
    

    f.legend(title='markers based on mask')
    f.show()
    
from typing import Iterator, List, Tuple

def appliance_augmentator(dataset:Dict[str,list], sample_length:int, n_appliances_per_sample:int, random_state:int=None) -> Iterator[Tuple[np.ndarray, np.ndarray, List[str]]]:
    
    # Initialize seeded random number generator
    rng = np.random.default_rng(seed=random_state)
    
    # How many appliances are mixed together
    N = n_appliances_per_sample
    
    # Sample length / Window size
    L = sample_length
    
    # Noise floor
    NOISE_FLOOR = 20.0
    
    # Get all available appliances
    appliance_names = tuple(dataset.keys())
    
    # How many appliances are there?
    n_appliances = len(appliance_names)
    print("Number of appliances: ", n_appliances)
    # Start endless generator of samples
    while True:
        # pre-allocate array for time series
        series = np.zeros(L, dtype=np.float64)
        
        # pre-allocate boolean array for masks
        labels = np.zeros((n_appliances, L), dtype=bool)
        
        # Select N random appliances (no replace, because same appliance should not appear twice in the same sample)
        # TODO: Relax this limitation in the future, for cases where multiple of same appliances are in the same household
        for appliance_idx in rng.choice(n_appliances, size=N, replace=False):
            appliance_name = appliance_names[appliance_idx]

            # Pick random sample of a selected appliance
            n_available_samples = len(dataset[appliance_name])
            sample_idx = rng.choice(n_available_samples)
            
            # retrieve sample as NumPy array with appropriate dimensions
            sample_series = dataset[appliance_name][sample_idx].iloc[:].to_numpy().squeeze(axis=-1)
            
            # If sample is too short (shorter than L), give padding on both sides.
            if len(sample_series) <= L:
                padding = L // 2
                sample_series = np.pad(sample_series, (padding, padding), mode='constant', constant_values=0)
            
            
            # The total length of sample time-series
            sample_len = len(sample_series)
            
            # Sanity check(s)
            assert sample_len >= L, f'Sample length should be equal or larger than L: {sample_len} >= {L}'
            
            sample_offset = rng.choice(sample_len - L)
            
            sample = sample_series[sample_offset:sample_offset+L]
            
            # TODO: Currently, we ignore device in idle state (x =< NOISE_FLOOR)
            mask = sample > NOISE_FLOOR  # find samples that are above noise floor
            
            series[:] += sample
            labels[appliance_idx, :] |= mask  # logical ORing the mask
            
        # Add random (constant) offset
        #series += rng.random() * (NOISE_FLOOR)
            
        # There has to be two samples present. Even though we combined two subsets, one or both could be empty.
        # Workaround until area of interest is implemented.
        if not (np.sum([(np.any(label) > 0) for label in labels]) == N):
            continue
            
        yield series, labels, appliance_names
        
        
        
import multiprocessing as mp



def process(random_state):
    import sys
    
    #""" ddddddd """
    # Generate Random seed:
    # Initialize internal seeded random number generator PRED EDITOM JE BIL SAMO rng = np.random.default.... in je bil edini v tej funkciji
    #random_state = np.random.randint(0, sys.maxsize)
    rng = np.random.default_rng(seed=random_state)
    #""" ddddddd """
    
    # Try to generate valid sample
    #while True:
    # pre-allocate array for time series
    series = np.zeros(L, dtype=np.float64)

    # pre-allocate boolean array for masks
    labels = np.zeros((n_appliances, L), dtype=bool)

    # Select N random appliances (no replace, because same appliance should not appear twice in the same sample)
    # TODO: Relax this limitation in the future, for cases where multiple of same appliances are in the same household
    # h
    for appliance_idx in rng.choice(n_appliances, size=N, replace=False):
        appliance_name = appliance_names[appliance_idx]
        #print(appliance_name)
        
        #""" ddddddd """
        # Generate Random seed:
        # Initialize internal seeded random number generator
        #random_state = np.random.randint(0, sys.maxsize)
        #rng = np.random.default_rng(seed=random_state)        
        #""" ddddddd """
        
        # Pick random sample of a selected appliance
        n_available_samples = len(dataset[appliance_name])
        sample_idx = rng.choice(n_available_samples)


        # retrieve sample as NumPy array with appropriate dimensions
        #print("appliance name: ", appliance_name, "sample_idx: ", sample_idx)
        try:
            sample_series = dataset[appliance_name][sample_idx].iloc[:].to_numpy().squeeze(axis=-1)
        except ValueError:
            #print("And the problematic devices is: ", appliance_name)
            continue

        # If sample is too short (shorter than L), give padding on both sides.
        if len(sample_series) <= L:
            padding = L // 2
            sample_series = np.pad(sample_series, (padding, padding), mode='constant', constant_values=0)

        ################## ADDED BY ANZE #######################
        # Check if any element of sample_series is above NOISE_FLOOR
        if not np.any(sample_series > NOISE_FLOOR):
            continue
        #########################################################
            
        # The total length of sample time-series
        sample_len = len(sample_series)

        # Sanity check(s)
        assert sample_len >= L, f'Sample length should be equal or larger than L: {sample_len} >= {L}'
        
        while True:
            
            #""" ddddddd """
            # Generate Random seed:
            # Initialize internal seeded random number generator
            #random_state = np.random.randint(0, sys.maxsize)
            #rng = np.random.default_rng(seed=random_state)        
            #""" ddddddd """
            
            
            sample_offset = rng.choice(sample_len - L)
            sample = sample_series[sample_offset:sample_offset+L]
            #print(sample)
            # TODO: Currently, we ignore device in idle state (x =< NOISE_FLOOR)
            mask = sample > NOISE_FLOOR  # find samples that are above noise floor
            
            if np.any(mask):
                break



        series[:] += sample
        labels[appliance_idx, :] |= mask  # logical ORing the mask

    # Add random (constant) offset
    #series += rng.random() * (NOISE_FLOOR)

    # There has to be two samples present. Even though we combined two subsets, one or both could be empty.
    # Workaround until area of interest is implemented.
    #if (np.sum([(np.any(label) > 0) for label in labels]) == N):
    #    break

    return series, labels, appliance_names
    
    

from tqdm import tqdm

def parallel_appliance_augmentator(_dataset:Dict[str,list], sample_length:int, n_appliances_per_sample:int, n_samples:int, random_state:int=None) -> list:
    import sys
    
    global N, L, NOISE_FLOOR, appliance_names, n_appliances, dataset
    
    dataset = _dataset

    # How many appliances are mixed together
    N = n_appliances_per_sample

    # Sample length / Window size
    L = sample_length

    # Noise floor
    NOISE_FLOOR = 20.0

    # Get all available appliances
    appliance_names = tuple(dataset.keys())

    # How many appliances are there?
    n_appliances = len(appliance_names)
    
    # if random_state is not defined, generate it on the fly
    if random_state is None:
        random_state = np.random.randint(0, sys.maxsize)
    
    with mp.Pool() as pool:
        outputs = list(tqdm(pool.imap(process, np.arange(n_samples) + random_state), total=n_samples))
        
    return tuple(outputs)

##############################################################################################    
##############################################################################################
##############################################################################################
##############################################################################################    
##############################################################################################
##############################################################################################
##############################################################################################    
##############################################################################################
##############################################################################################
##############################################################################################    
##############################################################################################
##############################################################################################
##############################################################################################    
##############################################################################################
##############################################################################################
##############################################################################################    
##############################################################################################
##############################################################################################    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
# MOJE    
##############################################################################################    
##############################################################################################
##############################################################################################
##############################################################################################    
##############################################################################################
##############################################################################################
##############################################################################################    
##############################################################################################
##############################################################################################
##############################################################################################    
##############################################################################################
##############################################################################################
##############################################################################################    
##############################################################################################
##############################################################################################
##############################################################################################    
##############################################################################################
##############################################################################################






##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
###################################### DATA MANIPULATION #####################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################

def ListAverage(lst):
    """
    This function returns the average number in the list
    
    Args:
        lst (list): the list for which we count the average number 
        
    Return:
        avg (int): average number calculated from numbers in the list
    """
    avg = sum(lst)/len(lst)
    return avg

def avg(lst):
    return sum(lst)/len(lst)

def ListReport(lst):
    """
    Args:
        lst (list): list that we want the report on
    Returns:
        report (dict): a dictionary which contains the smalles number, the average number and the biggest number in the list
    """
    report = {'min': min(lst), 'avg': ListAverage(lst), 'max': max(lst)}
    return report

def HousesList(dataset): 
    
    """
    Args:
        dataset (specified in print): 
    """
    
    print(f"type of dataset: {type(dataset)}")
    
    houses = []

    # We count the number of houses and make appropriate number of sublists in houses

    # To do that we first make a list of all devices from houses
    div_house_list = []
    for i in range(len(dataset)):
        div_house_list.append(dataset[i][0])
    div_house_list = sorted(div_house_list)
    #print(div_house_list)

    # Then we count the occurences 
    occurences = []
    for i in range(len(div_house_list)):
        if div_house_list.count(i) > 0: 
            occurences.append(div_house_list.count(i))
    #print(occurences)

    for h in range(len(occurences)):
        houses.append([])
        N = occurences[h]
    #    for i in range(N):
        for j in range(len(dataset)):
            #if datasets[j][0] == i: 
            #    houses.append([dataset[i]])
            if dataset[j][0] == h+1:
                houses[h].append(dataset[j])


    return houses



def WindowCounter(house: int, window_len: int, houses: list):
    """
    A function that counts the number of windows of the set length in the dataset
    
    Args:
        house (int): house number of the specific house we count windows for
        window_len (int): length of the window specified in HOURS!
        houses (list): a list of all houses and their data
    """
    # we initialize the list that will hold the maximal number of time windows of the chose length that can be extracted from the dataset
    lst = []
    
    #houses = HousesList(datasets)
    
    # We make a for loop that goes over all devices in the house and counts the maximal number of time windows and then adds them to lst
    for i in range(len(houses[house-1])):
        # Assuming the DataFrame is stored in houseX[0][2][0]
        window = pd.Timedelta(hours=window_len)
        start_time = houses[house-1][i][2][0].index[0]
        end_time = start_time + window

        count = 0

        while end_time <= houses[house-1][i][2][0].index[-1]:
            count += 1
            start_time = end_time
            end_time = start_time + window    

        lst.append(count)
        
    # All devices should have the same number of time windows, if not, we print a warning and choose the biggest number of windows that all devices can provide
    if len(set(lst)) != 1:
        print("Devices have different numbers of time windows, the biggest possible was chosen.")
        
    return min(lst)


def class_weights_tool(y_test: list):
    """
    This function returns class weights for each device in a prticular dataset in a form of a dicitionary.
    It does so by simply counting how many times the device is present througout the dataset.
    
    Args:
        y_test (list): the usual y_test part of the dataset, used in ML
        
    Returns: 
        class_weights_dictionary (dict): a dictionary which contains a number for each device which represents how many times the device appears in the dataset
    """
    # inspired by ronnie coleman
    light_weight, nums = [], []
    
    # this for loop goes over the collumns of the y_test dataset
    for j in range(0,len(y_test[0])):             
        
        count=0
        
        # gives 0,1,2,3,4,5,6....
        nums.append(j)
        
        # this loop goes over the rows in the y_test dataset
        for i in range(0,len(y_test)):
            
            # we count Trues in the whole column of y_test dataset
            if y_test[i][j] == True: count+=1     
        
        # we append Trues for the column of y_test dataset to the list light_weight
        light_weight.append(count)
        
    # makes the dictionary    
    class_weights_dictionary = dict(zip(nums, light_weight))             
    
    return class_weights_dictionary


def houses_function (dataset):
    
    # We initialize the list that will hold the data for all houses separately
    houses = []

    # We count the number of houses and make appropriate number of sublists in houses

    # To do that we first make a list of all devices from houses
    div_house_list = []
    for i in range(len(dataset)):
        div_house_list.append(dataset[i][0])
    div_house_list = sorted(div_house_list)
    #print(div_house_list)

    # Then we count the occurences 
    occurences = []
    for i in range(len(div_house_list)):
        if div_house_list.count(i) > 0: 
            occurences.append(div_house_list.count(i))
    #print(occurences)

    for h in range(len(occurences)):
        houses.append([])
        N = occurences[h]
    #    for i in range(N):
        for j in range(len(dataset)):
            #if datasets[j][0] == i: 
            #    houses.append([dataset[i]])
            if dataset[j][0] == h+1:
                houses[h].append(dataset[j])
                
    return houses


def XYDatasetGenerator(house: int, window_len: int, houses: list):
    
    """
    This function generates x_train, y_train, x_test, y_test, labels, which are slices of a natural dataset, of a specified length.
    
    Args:
        house (int): a number of the house in the dataset that we do this for
        window_len (int): a time window that represents the length of the slice, specified in HOURS!!!
        houses (list): a list of all houses and their data, we get it from HousesList
        
    Returns:
        x_train
        y_train
        x_test
        y_test
        labels
    """
    
    # We adjust house numbers to 0,1,2,3,4,... from 1,2,3,4,...
    print("house: ", house)
    house = house - 1
    print("house: ", house)
    
    # Noise floor
    NOISE_FLOOR = 20.0
    
    # We note the devices
    labels = []
    
    #houses = HousesList(dataset)
    
    for i in range(len(houses[house])):
        device = houses[house][i][1]
        if device == []:
            print("There is a nameless device or something thats not a device in the dataset, data that belongs to it wont be in the produced dataset.")
        else: 
            labels.append(device[0])
    
    labels = tuple(labels)
        
    #print("################################# labels ###############################")
    #print(labels)
    #print("#######################################################################")
    
    # We count the windows of the selected size
    number_of_windows = WindowCounter(house+1, window_len, houses)
    print(f"In house {house+1} there are {number_of_windows} {window_len}h long windows.")
    
    # We make a list of 3D numpy arrays of each device
    window_sum_devices_list = []
    window_sum_devices_3d = []
    list_window_array_3d = []
    for h in range(len(houses[house])):
        
        window = pd.Timedelta(hours=window_len)
        start_time = houses[house][0][2][0].index[0]
        end_time = start_time + window
        
        # Initialize an empty list to store the 2D numpy ndarrays
        window_array_list = []
        # Initialize a variable to keep track of the maximum number of samples in any 6-hour time window
        max_samples = 0
        
        for i in range(number_of_windows):
            # First we get pandas dataframe that is at the size of the window
            window_data = houses[house][h][2][0].loc[start_time:end_time]

            # We turn that dataframe into an array
            window_array = window_data.to_numpy(dtype=np.float64)

            # Find the number of samples in this 6-hour time window
            num_samples = window_array.shape[0]

            # Make a list of six_hour_arrays
            window_array_list.append(window_array)
            
            

            # Update the maximum number of samples
            max_samples = max(max_samples, num_samples)

            start_time = end_time
            end_time = start_time + window
            
        # time windows have sligthly different numbers of samples. 
        # Pad the 2D numpy ndarrays with zeros to ensure they all have the same number of samples
        for i in range(number_of_windows):
            window_array = window_array_list[i]
            num_samples = window_array.shape[0]
            if num_samples < max_samples:
                num_missing_samples = max_samples - num_samples
                zeros_array = np.zeros((num_missing_samples, window_array.shape[1]), dtype=np.float64)
                window_array_list[i] = np.vstack((window_array, zeros_array))
                
        # Shuffle the order of the 2D numpy ndarrays
        #np.random.shuffle(window_array_list)

        # Stack the 2D numpy ndarrays into a 3D numpy ndarray
        window_array_3d = np.stack(window_array_list)
        
        # Add the 3D numpy ndarray to the list
        list_window_array_3d.append(window_array_3d)
    
    #print(list_window_array_3d[1:])
    
    #print("|||||||||sum_devices|||||||||||||||||")
    #print(window_sum_devices_list[888])
    #print("|||||||||||||||||||||||||||||||||")
    
    # Shuffle the order of the 2D numpy ndarrays
    #np.random.shuffle(window_sum_devices_list)

    # Stack the 2D numpy ndarrays into a 3D numpy ndarray
    #window_sum_devices_3d = np.stack(window_sum_devices_list)
    
    #print("###############################################")
    #print(window_sum_devices_3d)
    #print("###############################################")
    
    
    
    # In order to get the signal that was the output of the whole house we must make a sum of signals that belong to individual devices
    #shelf = list_window_array_3d[1]
    #for arr in list_window_array_3d[2:]:
    #    result += arr
    window_sum_devices_3d = np.sum(list_window_array_3d[1:], axis=0)


    #print("################################# sum of devices ###############################")
    #print(window_sum_devices_3d)
    #print(window_sum_devices_3d.shape)
    #print("#######################################################################")
    
    
    # Now we have to make a mask which will give us the information on all active devices (so all devices with P > 20.0W)
    
    # We make a dummy array so that we dont change the src
    list_window_array_3d_dummy = list_window_array_3d[1:]
    
    # We initialize a list that will hold arrays of masks
    array_mask_list = []
    
    for arr in list_window_array_3d_dummy:
        
        # Check if elements are greater than 20
        mask = arr > NOISE_FLOOR
        
        # Replace the elements with 1 if they are bigger than 20 and 0 if they arent
        arr[mask] = True
        arr[~mask] = False
        
        # This code would replace it with True and False instead of 1 and 0
        #arr = np.where(mask, True, False)
        
        # Add the resultin array to a list
        array_mask_list.append(arr)

        
    # Now we have to make an array that will tell us this data for all devices at once and not just for one 
    
    # Initialize an empty list to hold the results
    list_nonzero = []

    # Loop over each 3D array
    for arr in array_mask_list:
        
        # Check if any element in each 2D array is nonzero
        arr_nonzero = np.any(arr != 0, axis=1).astype(bool)
        #arr_nonzero = np.any(arr != False, axis=1)
        
        # Append the result to the list
        list_nonzero.append(arr_nonzero)

    # Concatenate the results into a single 2D array
    window_mask_2d = np.concatenate(list_nonzero, axis=1)#.T

    
    #print("################################# array_mask_list ###############################")
    #print(window_mask_2d)
    #print(window_mask_2d.shape)
    #print("#######################################################################")
    
    
    # Now we have to randomly mix the data, but we must be careful to match the time window in window_sum_devices_3d with the devices on/off status in window_mask_2d
    
    # First we generate a random permuation of indices
    #permutation = np.random.permutation(number_of_windows)
    
    # And then we apply them to the arrays
    #window_sum_devices_3d_shuffled = window_sum_devices_3d[permutation]
    #window_mask_2d_shuffled = window_mask_2d[permutation]
    
    
    #print("################################# array_mask_list ###############################")
    #print(window_mask_2d_shuffled)
    #print(window_mask_2d_shuffled.shape)
    #print(permutation)
    #print("#######################################################################")
    
    
    # Now we just have to split the arrays into x_train, y_train, x_test, y_test
    
    # get the split index
    split_idx = int(number_of_windows * 0.8)

    # shuffle the indices
    indices = np.arange(number_of_windows)
    np.random.shuffle(indices)

    # split the indices into train and test sets
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]

    # use the indices to split the arrays
    x_train = window_sum_devices_3d[train_indices]
    x_test = window_sum_devices_3d[test_indices]
    y_train = window_mask_2d[train_indices]
    y_test = window_mask_2d[test_indices]

    # check the shapes
    print("The produced datasets have the following shapes:")
    print("x_train: ", x_train.shape)  # (1975, 3759, 1)
    print("x_test:  ", x_test.shape)   # (494, 3759, 1)
    print("y_train: ", y_train.shape)  # (1975, 9)
    print("y_test:  ", y_test.shape)   # (494, 9)
    
    
    return x_train, y_train, x_test, y_test, labels




def MaxSamples(house: int, window_len: int, houses: list):
    
    """
    This function finds the max number of samples in slices of a natural dataset, of a specified length.
    
    Args:
        house (int): a number of the house in the dataset that we do this for
        window_len (int): a time window that represents the length of the slice, specified in HOURS!!!
        houses (list): a list of all houses and their data, we get it from HousesList
        
    Returns:
        max_samples (int): number of samles that appear in the time window
    """
    
    # We adjust house numbers to 0,1,2,3,4,... from 1,2,3,4,...
    house = house - 1
    #print("house: ", house)
    
    # We count the windows of the selected size
    number_of_windows = WindowCounter(house+1, window_len, houses)
    #print(f"In house {house+1} there are {number_of_windows} {window_len}h long windows.")
    
    # We make a list of 3D numpy arrays of each device
    window_sum_devices_list = []
    window_sum_devices_3d = []
    list_window_array_3d = []
    for h in range(len(houses[house])):
        
        window = pd.Timedelta(hours=window_len)
        start_time = houses[house][0][2][0].index[0]
        end_time = start_time + window
        
        # Initialize an empty list to store the 2D numpy ndarrays
        window_array_list = []
        # Initialize a variable to keep track of the maximum number of samples in any 6-hour time window
        max_samples = 0
        
        for i in range(number_of_windows):
            # First we get pandas dataframe that is at the size of the window
            try:
                window_data = houses[house][h][2][0].loc[start_time:end_time]
            except KeyError as e:
                pass
            # We turn that dataframe into an array
            window_array = window_data.to_numpy(dtype=np.float64)

            # Find the number of samples in this 6-hour time window
            num_samples = window_array.shape[0]

            # Make a list of six_hour_arrays
            window_array_list.append(window_array)
            
            

            # Update the maximum number of samples
            max_samples = max(max_samples, num_samples)

            start_time = end_time
            end_time = start_time + window
    
    return max_samples


    
    
#x_train, y_train, x_test, y_test, labels = XYDatasetGenerator(2,6)

def data_preparation(dataset):
    X = defaultdict(lambda: [])

    for (idx, appliances, data, good_sections) in dataset:
        if not appliances:
            continue
            
        appliance = appliances[0]
        data = data[0]
    
        samples = [data[good.start:good.end] for good in good_sections]
        X[appliance].extend(samples)
        
    for appliance, samples in X.items():
        print(appliance, len(samples))
        
    return X


def DevicesDataXY(number_of_datasets: int, 
                  number_of_devices_in_datasets: int, 
                  number_of_all_devices: int):
    """
    Args:
        number_of_dataset (int): a number of datasets used, we use multiple datasets in the Dictlist to evaluate different randomly generated datasets
        number_of_devices_in_datasets (int): a number of devices in total (DiT) that will be present in the dataset
        number_of_all_devices (int): number of all devices that are available for the dataset
        
    Returns:
        devicesX_Y (list)
        dataX_Y (list)
    
    This function uses processed_data and shapes it into two lists 
    one of them contains data from devices (dataX_Y) and one of them names of devices (devicesX_Y).
    Lists contain multiple datasets specified with number_of_datasets, all of which have 
    the same number of devices in them, specified by number_of_devices_in_datasets.
    We choose devices for datasets randomly, thus we have to specify the number of all devices in the REFIT (22) or UKDALE dataset (54).
    """
    # we extract the dictionary processed_data into a list AllTable
    AllTable = [[k,v] for k,v in processed_data.items()]
    
    devicesX_Y = []
    dataX_Y = []
    
    # for loop goes over all datasets
    for i in range(0,number_of_datasets):    
        devicesY = []
        dataY = []
        j = 0
        
        # while loop goes over all devices in datasets
        while j < number_of_devices_in_datasets:
            
            # we get a random number with random library
            random_number = random.randrange(number_of_all_devices)
            
            # we use the random number to get a random device and random data that belongs to it from AllTable
            random_device = AllTable[random_number][0]
            random_data = AllTable[random_number][1]
            
            # we append the device and its data if it doesn't already exist 
            # and thus avoid having the same device more then once in the same dataset
            if random_device not in devicesY:
                devicesY.append(random_device)
                dataY.append(random_data)
                j += 1
        
        devicesX_Y.append(devicesY)
        dataX_Y.append(dataY)
    
    return devicesX_Y, dataX_Y


def DevicesDataHoly5(dataset_name: list):
    
    """
    This function does the same as DevicesDataXY except it makes only one dataset
    and the dataset doesn't have randomly selected devices, but it has particular 5 devices:
    Fridge, Washing Machine, Dish Washer, Microwave, Kettle.
    """
    
    # we extract the dictionary processed_data into a list AllTable
    AllTable=[[k,v] for k,v in processed_data.items()]
    
    devicesX_Y, dataX_Y, devicesY, dataY = [], [], [], []
    
    # the holy 5 devices are in this order in our processed_data from refit
    if dataset_name == "refit": nm_div = [0,3,4,11,12]
    
    # the holy 5 devices are in this order in our processed_data from uk-dale
    elif dataset_name == "uk-dale": nm_div = [42,41,4,11,8]
    
    else: 
        print("Wrong name of the dataset, write either refit or uk-dale") 
    
    # we go over the AllTable and append devices and data for the holy 5 devices
    for i in range(5):
        number = nm_div[i]
        device = AllTable[number][0]
        data = AllTable[number][1]
        devicesY.append(device)
        dataY.append(data)
    
    # we make a useless step because it works better with other functions that way
    devicesX_Y.append(devicesY)
    dataX_Y.append(dataY)
    
    return devicesX_Y, dataX_Y


def ListsToDictlist(devices: list, data: list, number_of_devices: int):
    
    """
    This function takes lists from DevicesDataXY or DevicesDataHoly5 and
    turns each of the sublists into a dictionary and then appends those dicitonaries into a list. 
    
    Args:
        devices (list): a list of devices {devicesX_Y}
        data (list): a list of data from devices {dataX_Y}
        number_of_devices (int): 
        
    Returns:
        Dictlist (dict): a dictionary of lists
    
    """
    Dictlist=[]
    
    # we make dictionaries out of lists that we input and append them to list of dictionaries (Dictlist)
    for i in range(0,number_of_devices):
        
        dictionary = dict(zip(devices[i], data[i]))
        Dictlist.append(dictionary)
        
    return Dictlist


def Generate4(Dictlist: list, 
              sample_length: int, 
              n_appliances_per_sample: int, 
              dataset_number: int):
    """
    Args:
        Dictlist (list): a list of dicitionaries generated by function ListsToDictlist
        sample_length (int): length of samples aka. length of the timewindow, we use 2550 as proposed and explained by Tanoni et al.
        dataset_number (int): number of the dataset, we have multiple datasets in the Dictlist to evaluate different randomly generated datasets
        n_appliances_per_sample (int): number of active devices (AD)

    Returns:
        x_train
        x_test
        y_train
        y_test
        labels
    """
    
    X, Y, labels = [], [], None
    
    generator = appliance_augmentator(Dictlist[dataset_number], 
                                      sample_length=sample_length, 
                                      n_appliances_per_sample=n_appliances_per_sample, 
                                      random_state=0xDEADBEEF)
    
    for idx, (_x, _y, labels) in zip(range(120_000), generator):
        X.append(_x), Y.append(_y)
        
    X, Y = np.asarray(X), np.asarray(Y)
    y = np.any(Y, axis=-1) > 0
    sample_weight = np.sum(Y, axis=-1) / 128
    
    # split into train test sets
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # adds the needed dimension
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    
    return x_train, x_test, y_train, y_test, labels


def Generate4Upgraded(Dictlist: list, 
              sample_length: int, 
              n_appliances_per_sample: int, 
              dataset_number: int,
              number_of_generated_samples: int
                     ):
    """
    This function does the same as Generate4 except here we have one more argument number_of_generated_samples,
    with which we can specify the number of all generated samples. 
    """
    
    X, Y, labels = [], [], None
    
    generator = parallel_appliance_augmentator(Dictlist[dataset_number], 
                                      sample_length=sample_length, 
                                      n_appliances_per_sample=n_appliances_per_sample,
                                      n_samples=number_of_generated_samples,
                                      random_state=0xAAAAA)  # It doesnt work with DEADBEEF for some reason 
    
    for idx, (_x, _y, labels) in zip(range(number_of_generated_samples), generator):
        X.append(_x), Y.append(_y)
        
    X, Y = np.asarray(X), np.asarray(Y)
    y = np.any(Y, axis=-1) > 0
    sample_weight = np.sum(Y, axis=-1) / 128
    
    # split into train test sets
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # add the needed dimension
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    
    return x_train, x_test, y_train, y_test, labels




def GenerateRandom4(Dictlist: list, 
              sample_length: int,  
              dataset_number: int,
              table_of_options: list,
              number_of_generated_samples: int
              ):
    """
    Args:
        Dictlist (list): a list of dicitionaries generated by function ListsToDictlist
        sample_length (int): length of samples aka. length of the timewindow, we use 2550 as proposed and explained by Tanoni et al.
        dataset_number (int): number of the dataset, we have multiple datasets in the Dictlist to evaluate different randomly generated datasets
        table_of_options (list): a list of possible active devices (AD)
        number_of_generated_samples (int): a number of samples that we generate, we use 120_000
    
    Returns:
        x_train
        x_test
        y_train
        y_test
        labels
        
        These datasets consist of equal parts of options from table of options.
        EX: If you have table_of_options = [1,2,3,4] the dataset will be 1/4 data with 1 active device, 1/4 data with 2 active devices, ...
        Dataset is mixed so that samples are randomly dispersed throughout the dataset
    """
    
    from tqdm import tqdm
    from multiprocessing import Pool
    
    print('Generating datasets with variable active devices ...')
    
    # First we calculate the number of samples that needs to be generated for each of the datasets that will be mixed into one
    number_of_generated_samples = number_of_generated_samples / len(table_of_options)
    number_of_generated_samples = int(number_of_generated_samples)
    
    # we introduce np.arrays in appropriate shape by making them as the first dataset with table_of_options[0] active devices
    print(f"{table_of_options[0]} AD:")
    x_tr, x_te, y_tr, y_te, labels = Generate4Upgraded(Dictlist,
                                                        sample_length,
                                                        table_of_options[0],
                                                        dataset_number,
                                                        number_of_generated_samples)
    
    # we rename them for later use if len(table_of_options) > 1
    x_train = x_tr
    x_test = x_te
    y_train = y_tr
    y_test = y_te
    
    #dolzina = len(table_of_options)
    #ananas1 = 0
    #ananas2 = 0
    #for i in range(dolzina):
    #    ananas2 += 1
    #    if ananas2 % 5 == 0:
    #        table_of_options.insert(i, 0)
    #        ananas1 += 1
    #        
    #print(table_of_options)    
    # 
    ## we make the rest of the datasets and append them to the first one
    for i in range(1,len(table_of_options)):
        print(f"{i} AD:")
    #    if table_of_options[i] == 0: continue
        n_appliances_per_sample = table_of_options[i]
        x_tr, x_te, y_tr, y_te, lab = Generate4Upgraded(Dictlist,
                                                        sample_length,
                                                        n_appliances_per_sample,
                                                        dataset_number,
                                                        number_of_generated_samples)
        x_train = np.append(x_train, x_tr, axis=0)
        x_test = np.append(x_test, x_te, axis=0)
        y_train = np.append(y_train, y_tr, axis=0)
        y_test = np.append(y_test, y_te, axis=0)
        
    # we print out the final shapes before returning new datasets
    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
        
    
    # Now that we have a dataset that consists of len(table_of_options) datasets with different numbers of active devices,
    # we just have to make the final dataset that will be made of exactly the same samples but in a random order.
    
    # To do so first introduce new tables  
    tm_train, tm_test = [], []
    
    # We adjust the number_of_generated_samples back to its original value
    number_of_generated_samples = int(number_of_generated_samples*len(table_of_options))
    #print("number_of_generated_samples: ", number_of_generated_samples)
    
    # we fill tables with integers in order 1,2,3,4,5,6...
    
    #for m in range(1,int(number_of_generated_samples*0.8)):
    #print("range(1,",int(math.floor((number_of_generated_samples*0.8)/len(table_of_options))*len(table_of_options)-1),")")
    for m in range(1,int(math.floor((number_of_generated_samples*0.8)/len(table_of_options))*len(table_of_options)-1)):
        tm_train.append(m)
    
    #for n in range(1,int(number_of_generated_samples*0.2)):
    #print("range(1,",int(math.floor((number_of_generated_samples*0.2)/len(table_of_options))*len(table_of_options)-1),")")
    for n in range(1,int(math.floor((number_of_generated_samples*0.2)/len(table_of_options))*len(table_of_options)-1)):
        tm_test.append(n)
    
    # we shuffle the order of those integers so that it is random
    random.shuffle(tm_train)
    random.shuffle(tm_test)
    
    
    """
    # we introduce np.arrays in an appropriate shape by assigning the first sample of 
    # the final datasets to be the first samples of the first datasest
    x_train_random = np.expand_dims(x_train[0], axis=0)
    x_test_random = np.expand_dims(x_test[0], axis=0)
    y_train_random = np.expand_dims(y_train[0], axis=0)
    y_test_random = np.expand_dims(y_test[0], axis=0)
    
    
    # we append to the arrays by using random order of tm_train
    # to make x and y array in the same random order (because we move the same random row in x and y train dataset)
    
     
    # we also split x : y = 80 : 20
    #for g in range(0,int(number_of_generated_samples*0.8-1)):
    #print("range(0,",int(number_of_generated_samples*0.8-1),")")
    #print("range(0,",int(math.floor((number_of_generated_samples*0.8)/len(table_of_options))*len(table_of_options)-3),")")
    for g in tqdm(range(0,int(math.floor((number_of_generated_samples*0.8)/len(table_of_options))*len(table_of_options)-3))):
        
        # we add the needed dimension ex: (256, 1) -> (1, 256, 1)
        x_applicable = np.expand_dims(x_train[tm_train[g]], axis=0)
        y_applicable = np.expand_dims(y_train[tm_train[g]], axis=0)
        
        # we append the applicable lists to the dataset
        x_train_random = np.append(x_train_random, x_applicable, axis=0)
        y_train_random = np.append(y_train_random, y_applicable, axis=0)

    
    # we append to the arrays by using random order of tm_test
    # to make the array in the same random order    
    #for h in range(0,int(number_of_generated_samples*0.2-1)): 
    #print("range(0,",int(math.floor((number_of_generated_samples*0.2)/len(table_of_options))*len(table_of_options)-3),")")
    for h in tqdm(range(0,int(math.floor((number_of_generated_samples*0.2)/len(table_of_options))*len(table_of_options)-3))): 
        
        # we add the needed dimension ex: (256, 1) -> (1, 256, 1)
        x_app = np.expand_dims(x_test[tm_test[h]],axis=0)
        y_app = np.expand_dims(y_test[tm_test[h]],axis=0)
        
        # we append the applicable lists to the dataset
        x_test_random = np.append(x_test_random, x_app, axis=0)
        y_test_random = np.append(y_test_random, y_app, axis=0)
    """    
    #from numba import njit
    #@njit
    #def NumbaConcat(x_random, y_random, x, y, tm, index):
    #    x_applicable = np.expand_dims(x[tm[index]], axis = 0)
    #    y_applicable = np.expand_dims(y[tm[index]], axis = 0)
    #    x_random = np.concatenate((x_random, x_applicable), axis = 0)
    #    y_random = np.concatenate((x_random, x_applicable), axis = 0)
    #    return x_random, y_random
    
    # we introduce np.arrays in an appropriate shape
    x_train_random = np.empty((0, sample_length, 1))
    y_train_random = np.empty((0, len(Dictlist[dataset_number])))
    x_test_random = np.empty((0, sample_length, 1))
    y_test_random = np.empty((0, len(Dictlist[dataset_number])))
    
    # we concatenate to the arrays by using random order of tm_train
    # to make x and y array in the same random order (because we move the same random row in x and y train dataset)
    # we split x : y = 80 : 20 like usual
    
    for g in tqdm(range(0,int(math.floor((number_of_generated_samples*0.8)/len(table_of_options))*len(table_of_options)-3))):
        x_applicable = np.expand_dims(x_train[tm_train[g]], axis=0)
        y_applicable = np.expand_dims(y_train[tm_train[g]], axis=0)
        x_train_random = np.concatenate((x_train_random, x_applicable), axis=0)
        y_train_random = np.concatenate((y_train_random, y_applicable), axis=0)
        #x_train_random, y_train_random = NumbaConcat(x_train_random, y_train_random, x_train, y_train, tm_train, g)
    for h in tqdm(range(0,int(math.floor((number_of_generated_samples*0.2)/len(table_of_options))*len(table_of_options)-3))):
        x_app = np.expand_dims(x_test[tm_test[h]],axis=0)
        y_app = np.expand_dims(y_test[tm_test[h]],axis=0)
        x_test_random = np.concatenate((x_test_random, x_app), axis=0)
        y_test_random = np.concatenate((y_test_random, y_app), axis=0)
        #x_test_random, y_test_random = NumbaConcat(x_test_random, y_test_random, x_test, y_test, tm_test, h)
    """
    # Define the number of parts
    num_parts = 100

    # Calculate the number of iterations for each part
    iterations_per_part_1 = int(math.floor((number_of_generated_samples * 0.8) / len(table_of_options)) * len(table_of_options) - 3) // num_parts
    iterations_per_part_2 = int(math.floor((number_of_generated_samples * 0.2) / len(table_of_options)) * len(table_of_options) - 3) // num_parts

    # Initialize the variables for storing the results
    #x_train_random = []
    #y_train_random = []
    #x_test_random = []
    #y_test_random = []

    # Process the first loop in parts
    for part in tqdm(range(num_parts)):
        start_index = part * iterations_per_part_1
        end_index = (part + 1) * iterations_per_part_1

        for g in range(start_index, end_index):
            x_applicable = np.expand_dims(x_train[tm_train[g]], axis=0)
            y_applicable = np.expand_dims(y_train[tm_train[g]], axis=0)
            x_train_random = np.concatenate((x_train_random, x_applicable), axis=0)
            y_train_random = np.concatenate((y_train_random, y_applicable), axis=0)

    # Process the second loop in parts
    for part in tqdm(range(num_parts)):
        start_index = part * iterations_per_part_2
        end_index = (part + 1) * iterations_per_part_2

        for h in range(start_index, end_index):
            x_app = np.expand_dims(x_test[tm_test[h]], axis=0)
            y_app = np.expand_dims(y_test[tm_test[h]], axis=0)
            x_test_random = np.concatenate((x_test_random, x_app), axis=0)
            y_test_random = np.concatenate((y_test_random, y_app), axis=0)
    """
    
    
    # we print out the final shapes before returning new datasets
    print(x_train_random.shape, x_test_random.shape, y_train_random.shape, y_test_random.shape)
    
    return x_train_random, x_test_random, y_train_random, y_test_random, labels



def DatasetMerger(pickle_names, path_to_pickles):
    """
    Merges datasets from pickle files into one dictionary. Also deletes dish washer from REFIT because its bad for some reason.

    Args:
        pickle_names (list of str): List of pickle file names to merge.
        path_to_pickles (str): Path to the folder in which there are processed datasets saved as pkl

    Returns:
        dict: Merged dataset as a dictionary
    """
    # Read in each pickle file as a dictionary and store them in a list
    data_list = []
 
    for pickle_name in pickle_names:
        data = pickle.load(open(f'{path_to_pickles}/{pickle_name}', 'rb'))
        if pickle_name == 'REF_processed.pkl': 
            del data['dish washer']
            print('Dish washer deleted in REFIT, because its bad.')
        data_list.append(data)

    # Merge the dictionaries into a single dictionary
    merged_data = {}
    for data in data_list:
        for key in data:
            if key not in merged_data:
                merged_data[key] = []
            merged_data[key].extend(data[key])

    print("Datasets merged successfully :)")
    
    return merged_data


def FilterDevices (merged_dataset: dict, wanted_devices: list):
    
    """
    This function filters the devices in the merged dataset, so that we are left with only the ones we want.
    
    Args: 
        merged_dataset (dict): a dataset merged from multiple NILM datasets, we get it with DatasetMerger
        wanted_devices (list): a list of the devices we want in the dataset
    Returns: 
        filtered_merged_dataset (dict): a dataset that only has the devices we want, the devices that the house we evaluate on has
    """
    
    filtered_merged_dataset = merged_dataset.copy()

    # We filter out any devices that are not on the wanted_devices list
    
    for i, j in merged_dataset.items():
        if i not in wanted_devices:
            del filtered_merged_dataset[i]

    # We fix some specific things i guess??? (not my code)
            
    if 'dish washer' in filtered_merged_dataset.keys():
        filtered_merged_dataset['dishwasher'] = filtered_merged_dataset.pop('dish washer')

    if 'electric oven' in filtered_merged_dataset.keys():
        filtered_merged_dataset['oven'] = filtered_merged_dataset.pop('electric oven')
        
    # We sort the devices in the correct order
    sorted_dict = {k: v for k, v in sorted(filtered_merged_dataset.items(), key=lambda item: wanted_devices.index(item[0]))}
        
    print("Devices filtered successfully :)")
        
    return sorted_dict




def SmartFilterDevices(merged_dataset: dict, wanted_devices: list):
    from fuzzywuzzy import fuzz
    from collections import Counter, OrderedDict
    
    """
    This function works just like FilterDevices, but it uses a smart way of matching devices.
    
    Args: 
        merged_dataset (dict): a dataset merged from multiple NILM datasets, we get it with DatasetMerger
        wanted_devices (list): a list of the devices we want in the dataset
    Returns: 
        filtered_merged_dataset (dict): a dataset that only has the devices we want, the devices that the house we evaluate on has
    """
    
    filtered_merged_dataset = merged_dataset.copy()
    

    # We filter out any devices that are not on the wanted_devices list
    matched_devices = []
    to_delete = []

    for device in wanted_devices:
        match = max(filtered_merged_dataset, key=lambda x: fuzz.token_set_ratio(device, x))
        print("device: ", device)
        if fuzz.token_set_ratio(device, match) > 85:
            matched_devices.append(match)
            print("match: ", match)
            print(" ")

    
    # There can be multiple of the same devices, so we count them 
    device_count_list = Counter(matched_devices)
    
    # Make final_dict that has all the devices and their data
    final_dict = {}
    filtered_merged_dataset_clone = filtered_merged_dataset.copy()
    for device, data in filtered_merged_dataset_clone.items():
        for i in range(device_count_list[f'{device}']):
            final_dict[f"{device} {i+1}"] = data
            
    # Make sorted_dict out of the final_dict, which is sorted like it should be
    sorted_dict = OrderedDict((key, final_dict[key]) for device in matched_devices for key in final_dict.keys() if device in key)
    print(" ")
    print(" ")
    print("sorted_dict: ")
    for k, v in sorted_dict.items():
        print(k, len(v))
    print(" ")
    print("Devices filtered successfully :)")
     
    return sorted_dict


def SmarterFilterDevices(merged_dataset: dict, wanted_devices: list):
    from fuzzywuzzy import fuzz
    from collections import Counter, OrderedDict
    import spacy
    
    """
    This function works just like FilterDevices, but it uses a smart way of matching devices.
    
    Args: 
        merged_dataset (dict): a dataset merged from multiple NILM datasets, we get it with DatasetMerger
        wanted_devices (list): a list of the devices we want in the dataset
    Returns: 
        filtered_merged_dataset (dict): a dataset that only has the devices we want, the devices that the house we evaluate on has
    """
    
    print('Filtering devices ...')
    
    filtered_merged_dataset = merged_dataset.copy()
    
    nlp = spacy.load("en_core_web_lg")
    
    matched_devices = []

    all_devices = []
    for k, v in merged_dataset.items():
        all_devices.append(k)


    for wanted_device in wanted_devices:        

        # Find the device with the highest similarity score
        max_similarity = -1
        matched_device = ''
        for device in all_devices:
            similarity = nlp(device).similarity(nlp(wanted_device))
            if similarity > max_similarity:
                max_similarity = similarity
                matched_device = device

        matched_devices.append(matched_device)
        print(' ')
        print("wanted_device: ", wanted_device)
        print("matched_device: ", matched_device)
        print(' ')
    
    # There can be multiple of the same devices, so we count them 
    device_count_list = Counter(matched_devices)
    
    # Make final_dict that has all the devices and their data
    final_dict = {}
    filtered_merged_dataset_clone = filtered_merged_dataset.copy()
    for device, data in filtered_merged_dataset_clone.items():
        for i in range(device_count_list[f'{device}']):
            final_dict[f"{device} {i+1}"] = data
            
    # Make sorted_dict out of the final_dict, which is sorted like it should be
    sorted_dict = OrderedDict((key, final_dict[key]) for device in matched_devices for key in final_dict.keys() if device in key)
    print(" ")
    print(" ")
    print("sorted_dict: ")
    for k, v in sorted_dict.items():
        print(k, len(v))
    print(" ")
    print("Devices filtered successfully :)")
     
    return sorted_dict


def TrainxxyylEvalxxyyl (pickle_names: list, path_to_pickles: str, path_to_sliced_natural: str):
    """
    Whole data preparation. The only things you need are natural datasets and the sliced dataset.
    
    Args:
        pickle_names (list): Names of processed natural datasets
        path_to_pickles (str): Path to processed natural datasets
        wanted_devices (list): Devices we want in the merged dataset
        
    Returns:
        filtered_merged_dataset (dict): a dataset that was merged from the specified natural datasets and filtered for devices you want
    """
    
    print('Generating x_train, y_train, x_test, y_test, labels:')
    
    # We import the sliced natural dataset for house 8 in REFIT dataset
    h = pickle.load(open(f'{path_to_sliced_natural}', 'rb'))
    
    # We specify the wanted_devices from the house
    wanted_devices = h[4]
    
    print('Merging natural datasets ...')
    
    # Use function DatasetMerger from NUK
    merged_dataset = NUK.DatasetMerger(pickle_names, path_to_pickles)
    
    print('Merged successfully!')
    print(' ')

    print('Filtering the merged dataset ...')
    
    # Use the function FilterDevices
    filtered_merged_dataset = NUK.SmartFilterDevices(merged_dataset, wanted_devices)
    
    print('Filtered successfully!')
    print(' ')
    
    print('Basic information about the generated datasets:')
    
    # We will pretrain the model on all possible numbers of active devices so [0, 1, 2, ..., NumberOfAllDevices].
    table_of_options = [0]
    for i in range(len(h[3][0])):
        table_of_options.append(i + 1)
    print(f'Number of active devices can be: {table_of_options}')

    # We select the window size based on how many samples there are in an actual time_window of the house 8 in REFIT dataset.
    window_size = len(h[2][0])
    print(f'Window size is {window_size} samples')

    # We will only have one dataset this time, our function is built for a possibility of multiple. So we just say nm = 0
    nm = 0

    # We will train on 120_000 samples because that should be enough according to our experience
    nm_samples = 120_000
    print(f'We are training on {nm_samples} samples (hardcoded)')
    print(' ')

    # Now we simply have to make a Dictlist which will only have one dictionary.
    # This step has to be done just because of how my functions are made.
    Dictlist = []
    Dictlist.append(filtered_merged_dataset)

    # Finally we use GenerateRandom4 function and get the datasets
    x_train, x_test, y_train, y_test, labels = NUK.GenerateRandom4(Dictlist, window_size, nm, table_of_options, nm_samples)
    
    x_train_h, x_test_h, y_train_h, y_test_h, labels_h = h[0], h[1], h[2], h[3], h[4]
    
    return x_train, x_test, y_train, y_test, labels, x_train_h, x_test_h, y_train_h, y_test_h, labels_h



    
    
def LabelsValuesPredictionsToGraph (devices: list, values: list, predictions: list):
    
    """
    Plots two bar graphs on top of each other: one for labels and values, and the other for labels and predictions.
    Its done so that 
    
    Args:
    - devices (list): A list of strings representing device names.
    - values (list): A list of Boolean values indicating the status of each device.
    - predictions (list): A list of Boolean values indicating the predicted status of each device.
    """
    
    print('Graphs from data and prediction: ')
    
    true_values = [i for i in range(len(values)) if values[i]]
    false_values = [i for i in range(len(values)) if not values[i]]

    fig, ax = plt.subplots(figsize = (20, 2))
    ax.bar(true_values, [1]*len(true_values), color='green', label='ON')
    ax.bar(false_values, [1]*len(false_values), color='red', label='OFF')

    ax.set_xticks(range(len(devices)))
    ax.set_xticklabels([])

    ax.set_ylabel('DATA', fontsize = 16)
    ax.set_yticks([0, 1])
    ax.set_yticklabels([' ', ' '])

    ax.legend()
    plt.show()

    
    true_values = [i for i in range(len(predictions)) if predictions[i]]
    false_values = [i for i in range(len(predictions)) if not predictions[i]]

    fig, ax = plt.subplots(figsize = (20, 2))
    ax.bar(true_values, [1]*len(true_values), color='green', label='ON')
    ax.bar(false_values, [1]*len(false_values), color='red', label='OFF')

    ax.set_xticks(range(len(devices)))
    ax.set_xticklabels(devices, rotation=45, ha='right')

    ax.set_ylabel('PREDICTION', fontsize = 16)
    ax.set_yticks([0, 1])
    ax.set_yticklabels([' ', ' '])

    #ax.legend()
    plt.show()
    
    print(' ')
    
    
    
    
def ProcessMergedDataset (data):
    # Create a new dictionary to store the processed data
    processed_data = {}

    # Loop over the original dictionary
    for item in data:
        # Check if the item has fewer than 10 keys
        if len(data[item]) < 10:
            # Skip this item if it has fewer than 10 keys
            #print(item, len(data[item]))
            continue

        # Check if the item has a number at the end of its name
        if item[-1].isdigit():
            # If the item has a number at the end of its name, remove the number
            base_item = item.rstrip('0123456789')
        else:
            # Otherwise, use the item name as the base item
            base_item = item

        # Add the item's keys to the appropriate base item in the processed data dictionary
        if base_item in processed_data:
            processed_data[base_item] += data[item]
        else:
            processed_data[base_item] = data[item]

    # Return the processed data dictionary
    return processed_data



def MergeSimilarDevices(d: Dict[str, int], similarity_threshold: float) -> Dict[str, List[int]]:
    """
    Merges similar items in a dictionary based on Levenshtein distance similarity measure.

    Args:
        d (Dict[str, int]): The dictionary of items to merge.
        similarity_threshold (float): The similarity threshold above which items will be merged.

    Returns:
        Dict[str, List[int]]: The merged dictionary of items.
    """
    import spacy
    nlp = spacy.load("en_core_web_lg")
    
    # Convert the dictionary to a list of tuples
    items = list(d.items())
    
    # Initialize an empty dictionary to store the merged items
    merged_dict = {}
            
    # We loop over all devices in the dictionary
    already_processed = []
    for device, values in d.items():
        
        if device not in already_processed:
            # Initialize an empty list to store similar items
            similar_devices = []

            # We loop over all other devices in the dictionary by looping over all of them and then excluding device 
            # that we are processing as device with an if statement
            for other_device, values in d.items():

                if other_device != device:

                    # Calculate the Levenshtein distance between the two items
                    distance = nlp(device).similarity(nlp(other_device))

                    # If the distance is above the similarity threshold, add the item to the similar items list
                    if distance >= similarity_threshold:
                        similar_devices.append(other_device)#, count2))    

            if similar_devices:
                
                #print('Some devices will be merged: ', similar_devices)

                # Merge the dataframes for each key into a single list
                merged_list = []
                for device in similar_devices:
                    merged_list.extend(d[device])

                # Create a new key with the merged list of dataframes as its value
                new_device = ', '.join(similar_devices)
                merged_dict[new_device] = merged_list

                # put them on the already processed list
                for device in similar_devices:
                    already_processed.append(device)

            else:
                #print('This device will stay as is: ', [device])
                merged_dict[device] = d[device]  
                already_processed.append(device)
            
    return merged_dict





def BadDeviceDetector(d: dict, NOISE_FLOOR):

    for device, values in d.items():
       
        # Pick random sample of a selected appliance
        n_available_values = len(values)
        
        count = 0
        
        for value_idx in range(n_available_values):
            
            #print('device: ', device, 'value: ', value)
            
            sample_series = d[device][value_idx].iloc[:].to_numpy().squeeze(axis=-1)
            
            # Check if any element of sample_series is above NOISE_FLOOR
            if not np.any(sample_series > NOISE_FLOOR):
                count += 1
                continue
        
        if not count < n_available_values:
            print(f'{device} is bad!')

#    return 


def Mediator(labels: list, labels_h: list, y_pred_tf: list):
    #from fuzzywuzzy import fuzz
    from collections import Counter, OrderedDict
    import spacy
    
    """
    This function takes out the devices that are similar out of the ones in the bigger dataset. You use it so that you have the same devices in y_pred as you do in y_test from another dataset.
    
    Args: 
        labels (list): labels from the merged dataset
        labels_h (list): labels from the targeted dataset
        y_pred_tf (list): y_pred_tf from the merged dataset that has many devices
    Returns: 
        y_pred_tf_filtered (list): list that will have true and falses only for the devices that exist in the targeted dataset
    """
    
    print('Filtering devices ...')
    
    nlp = spacy.load("en_core_web_lg")
    
    matched_devices = []

    all_devices = []
    for app in labels: all_devices.append(app)
    wanted_devices = []
    for app in labels_h: wanted_devices.append(app)


    for wanted_device in wanted_devices:        

        # Find the device with the highest similarity score
        max_similarity = -1
        matched_device = ''
        for device in all_devices:
            similarity = nlp(device).similarity(nlp(wanted_device))
            if similarity > max_similarity:
                max_similarity = similarity
                matched_device = device

        matched_devices.append(matched_device)
        print(' ')
        print("wanted_device: ", wanted_device)
        print("matched_device: ", matched_device)
        print(' ')
    
    matched_devices = tuple(matched_devices)

    # Make final_dict that has all the devices and their data
    y_pred_tf_filtered = [[]] * len(y_pred_tf)
    for i in range(len(y_pred_tf)): y_pred_tf_filtered[i] = [y_pred_tf[i][labels.index(device)] for device in matched_devices]

    print("Devices filtered successfully :)")
        
    return y_pred_tf_filtered, matched_devices



def ClassificationReportToDF(report: str, class_name: str):
    """
    This function takes a classification report made by using sklearn and turns it into a df which is much nicer to use.
    The name of classes has to be specified, to suit the use case.
    
    Args:
        report (str): a classification report returned by sklearn
        class_name (str): a name of classes, EX: devices, fruits, vehicles, animals....
    Returns:
        report_df (pandas dataframe): a pandas dataframe of a report
    """

    # Convert the classification_report string to a pandas DataFrame
    report_df = pd.read_csv(StringIO(report), sep=r'\s{2,}', engine='python')
    # Reset the index to have the device names as a column
    report_df.reset_index(inplace=True)
    # Rename columns for better readability
    report_df.rename(columns={'index': class_name}, inplace=True)
    
    return report_df
    

##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################





##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
###################################### MODELS ################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
    
import tensorflow as tf
from tensorflow.keras import backend as K

def F1Score(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)

    precision = tp / (tp + fp + K.epsilon())
    recall = tp / (tp + fn + K.epsilon())
    f1 = 2 * precision * recall / (precision + recall + K.epsilon())
    # Handle the case where both precision and recall are zero
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)


def F1Score2(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)

    precision = tp / (tp + fp + 1e-5)
    recall = tp / (tp + fn + 1e-5)
    f1 = 2 * precision * recall / (precision + recall + 1e-5)

    return K.mean(f1)


def WeightedF1Score(class_weights):
    class_weights_dict = class_weights.copy()
    class_weights = list(class_weights_dict.values())
    class_weights = tf.cast(class_weights, tf.float32)
    def WeightedF1(y_true, y_pred):
        y_pred = K.round(y_pred)
        tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
        fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
        fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)

        precision = tp / (tp + fp + K.epsilon())
        recall = tp / (tp + fn + K.epsilon())
        f1 = 2 * precision * recall / (precision + recall + K.epsilon())
        # Handle the case where both precision and recall are zero
        f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
        weighted_f1 = K.sum(class_weights * f1) / K.sum(class_weights)

        return weighted_f1

    return WeightedF1
    
    
    
def VGG19_1D(classes,
             window_size,
             #gru,
             #gru_num,
             include_top=True,
             input_tensor=None,
              pooling=None):

    # Determine proper input shape
    input_shape = (window_size,1)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    # Block 1
    x = Conv1D(64, (3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Conv1D(64, (3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling1D((2), strides=(2), name='block1_pool')(x)

    # Block 2
    x = Conv1D(128, (3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv1D(128, (3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling1D((2), strides=(2), name='block2_pool')(x)

    # Block 3
    x = Conv1D(256, (3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv1D(256, (3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv1D(256, (3), activation='relu', padding='same', name='block3_conv3')(x)
    x = Conv1D(256, (3), activation='relu', padding='same', name='block3_conv4')(x)
    x = MaxPooling1D((2), strides=(2), name='block3_pool')(x)

    # Block 4
    x = Conv1D(512, (3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv1D(512, (3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv1D(512, (3), activation='relu', padding='same', name='block4_conv3')(x)
    x = Conv1D(512, (3), activation='relu', padding='same', name='block4_conv4')(x)
    x = MaxPooling1D((2), strides=(2), name='block4_pool')(x)

    # Block 5
    x = Conv1D(512, (3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv1D(512, (3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv1D(512, (3), activation='relu', padding='same', name='block5_conv3')(x)
    x = Conv1D(512, (3), activation='relu', padding='same', name='block5_conv4')(x)
    x = MaxPooling1D((2), strides=(2), name='block5_pool')(x)
    
    # GRU layer, btw GRU stands for Gated Recurrent Units
    #if gru == True:
    #    print("GRU enabled")
    #    x = GRU(gru_num, activation='tanh', recurrent_activation='sigmoid')(x)

    if include_top:
        # Classification block
        x = Flatten(name='flatten')(x)
        x = Dense(4096, activation='relu', name='fc1')(x)
        x = Dense(4096, activation='relu', name='fc2')(x)
        x = Dense(classes, activation='sigmoid', name='predictions')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling1D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling1D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='vgg19_1D')
    return model

def VGG11_1D(classes,
             window_size,
             include_top=True,
             input_tensor=None,
              pooling=None):

    # Determine proper input shape
    input_shape = (window_size,1)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    # Block 1
    x = Conv1D(64, (3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = MaxPooling1D((2), strides=(2), name='block1_pool')(x)

    # Block 2
    x = Conv1D(128, (3), activation='relu', padding='same', name='block2_conv1')(x)
    x = MaxPooling1D((2), strides=(2), name='block2_pool')(x)

    # Block 3
    x = Conv1D(256, (3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv1D(256, (3), activation='relu', padding='same', name='block3_conv2')(x)
    x = MaxPooling1D((2), strides=(2), name='block3_pool')(x)

    # Block 4
    x = Conv1D(512, (3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv1D(512, (3), activation='relu', padding='same', name='block4_conv2')(x)
    x = MaxPooling1D((2), strides=(2), name='block4_pool')(x)

    # Block 5
    x = Conv1D(512, (3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv1D(512, (3), activation='relu', padding='same', name='block5_conv2')(x)
    x = MaxPooling1D((2), strides=(2), name='block5_pool')(x)

    if include_top:
        # Classification block
        x = Flatten(name='flatten')(x)
        x = Dense(4096, activation='relu', name='fc1')(x)
        x = Dense(4096, activation='relu', name='fc2')(x)
        x = Dense(classes, activation='sigmoid', name='predictions')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling1D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling1D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='vgg11_1D')
    return model


def PE2(classes,
             window_size,
             method,
             method_num,
             include_top=True,
             input_tensor=None,
              pooling=None):

    # Determine proper input shape
    input_shape = (window_size,1)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    # Block 1
    x = Conv1D(64, (3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Conv1D(64, (3), activation='relu', padding='same', name='block1_conv2')(x)
    x = AveragePooling1D((2), strides=(2), name='block1_pool')(x)

    # Block 2
    x = Conv1D(128, (3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv1D(128, (3), activation='relu', padding='same', name='block2_conv2')(x)
    x = AveragePooling1D((2), strides=(2), name='block2_pool')(x)

    # Block 3
    x = Conv1D(256, (3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv1D(256, (3), activation='relu', padding='same', name='block3_conv2')(x)
    x = AveragePooling1D((2), strides=(2), name='block3_pool')(x)

    # Block 4
    x = Conv1D(512, (3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv1D(512, (3), activation='relu', padding='same', name='block4_conv2')(x)
    x = AveragePooling1D((2), strides=(2), name='block4_pool')(x)
    
    # Block 5
    x = Conv1DTranspose(512, (3), activation='relu', padding='same', name='block5_tran_conv1')(x)
    #x = Conv1DTranspose(512, (3), activation='relu', padding='same', name='block5_tran_conv2')(x)
    x = AveragePooling1D((2), strides=(2), name='block5_pool')(x)
    
    # GRU layer, btw GRU stands for Gated Recurrent Units; https://en.wikipedia.org/wiki/Gated_recurrent_unit
    if method == 'gru':
        print("GRU enabled")
        x = GRU(method_num, activation='tanh', recurrent_activation='sigmoid')(x)                
        
    if method == 'gru2':
        print("GRU enabled")
        x = GRU(method_num,activation='tanh',recurrent_activation='sigmoid',reset_after=True)(x)        
        
    if method == 'bigru':
        print("Bi-GRU enabled")
        x = Bidirectional(GRU(method_num, activation='tanh', recurrent_activation='sigmoid'))(x)
        
    if method == 'lstm':
        print("LSTM enabled")
        x = LSTM(method_num, activation='tanh', recurrent_activation='sigmoid')(x)
    
    # LSTM layer, btw LSTM stands for long short-term memory; https://en.wikipedia.org/wiki/Long_short-term_memory
    if include_top:
        # Classification block
        x = Flatten(name='flatten')(x)
        x = Dense(4096, activation='relu', name='fc1')(x)
        x = Dense(4096, activation='relu', name='fc2')(x)
        x = Dense(classes, activation='sigmoid', name='predictions')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling1D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling1D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='PE2')
    return model


def PEH(classes,
             window_size,
             method,
             method_num,
             include_top=True,
             input_tensor=None,
              pooling=None):

    # Determine proper input shape
    input_shape = (window_size,1)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    # Block 1
    x = Conv1D(64, (3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = AveragePooling1D((2), strides=(2), name='block1_pool')(x)

    # Block 2
    x = Conv1D(128, (3), activation='relu', padding='same', name='block2_conv1')(x)
    x = AveragePooling1D((2), strides=(2), name='block2_pool')(x)

    # Block 3
    x = Conv1D(256, (3), activation='relu', padding='same', name='block3_conv1')(x)
    x = AveragePooling1D((2), strides=(2), name='block3_pool')(x)

    # Block 4
    x = Conv1D(512, (3), activation='relu', padding='same', name='block4_conv1')(x)
    x = AveragePooling1D((2), strides=(2), name='block4_pool')(x)
    
    # Block 5
    x = Conv1DTranspose(512, (3), activation='relu', padding='same', name='block5_tran_conv1')(x)
    x = AveragePooling1D((2), strides=(2), name='block5_pool')(x)
    
    # GRU layer, btw GRU stands for Gated Recurrent Units; https://en.wikipedia.org/wiki/Gated_recurrent_unit
    if method == 'gru':
        print("GRU enabled")
        x = GRU(method_num, activation='tanh', recurrent_activation='sigmoid')(x)                
        
    if method == 'gru2':
        print("GRU enabled")
        x = GRU(method_num,activation='tanh',recurrent_activation='sigmoid',reset_after=True)(x)        
        
    if method == 'bigru':
        print("Bi-GRU enabled")
        x = Bidirectional(GRU(method_num, activation='tanh', recurrent_activation='sigmoid'))(x)
        
    if method == 'lstm':
        print("LSTM enabled")
        x = LSTM(method_num, activation='tanh', recurrent_activation='sigmoid')(x)
    
    # LSTM layer, btw LSTM stands for long short-term memory; https://en.wikipedia.org/wiki/Long_short-term_memory
    if include_top:
        # Classification block
        x = Flatten(name='flatten')(x)
        x = Dense(4096, activation='relu', name='fc1')(x)
        x = Dense(classes, activation='sigmoid', name='predictions')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling1D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling1D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='PEH')
    return model



def PEH2(classes,
             window_size,
             method,
             method_num,
             include_top=True,
             input_tensor=None,
              pooling=None):

    # Determine proper input shape
    input_shape = (window_size,1)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    # Block 1
    x = Conv1D(64, (3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Conv1D(32, (3), activation='relu', padding='same', name='block1_conv2')(x)
    #x = Conv1D(64, (3), activation='relu', padding='same', name='block1_conv2')(x)
    x = AveragePooling1D((2), strides=(2), name='block1_pool')(x)

    # Block 2
    x = Conv1D(128, (3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv1D(64, (3), activation='relu', padding='same', name='block2_conv2')(x)
    #x = Conv1D(128, (3), activation='relu', padding='same', name='block2_conv2')(x)
    x = AveragePooling1D((2), strides=(2), name='block2_pool')(x)

    # Block 3
    x = Conv1D(256, (3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv1D(128, (3), activation='relu', padding='same', name='block3_conv2')(x)
    x = AveragePooling1D((2), strides=(2), name='block3_pool')(x)

    # Block 4
    x = Conv1D(512, (3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv1D(256, (3), activation='relu', padding='same', name='block4_conv2')(x)
    x = AveragePooling1D((2), strides=(2), name='block4_pool')(x)
    
    # Block 5
    x = Conv1DTranspose(512, (3), activation='relu', padding='same', name='block5_tran_conv1')(x)
    #x = Conv1DTranspose(512, (3), activation='relu', padding='same', name='block5_tran_conv2')(x)
    x = AveragePooling1D((2), strides=(2), name='block5_pool')(x)
    
    # GRU layer, btw GRU stands for Gated Recurrent Units; https://en.wikipedia.org/wiki/Gated_recurrent_unit
    if method == 'gru':
        print("GRU enabled")
        x = GRU(method_num, activation='tanh', recurrent_activation='sigmoid')(x)                
        
    if method == 'gru2':
        print("GRU enabled")
        x = GRU(method_num,activation='tanh',recurrent_activation='sigmoid',reset_after=True)(x)        
        
    if method == 'bigru':
        print("Bi-GRU enabled")
        x = Bidirectional(GRU(method_num, activation='tanh', recurrent_activation='sigmoid'))(x)
        
    if method == 'lstm':
        print("LSTM enabled")
        x = LSTM(method_num, activation='tanh', recurrent_activation='sigmoid')(x)
    
    # LSTM layer, btw LSTM stands for long short-term memory; https://en.wikipedia.org/wiki/Long_short-term_memory
    if include_top:
        # Classification block
        x = Flatten(name='flatten')(x)
        x = Dense(4096, activation='relu', name='fc1')(x)
        x = Dense(4096, activation='relu', name='fc2')(x)
        x = Dense(4096, activation='relu', name='fc3')(x)
        x = Dense(classes, activation='sigmoid', name='predictions')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling1D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling1D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='PEH2')
    return model



def PC0(classes,
             window_size,
             method,
             method_num,
             k,
             include_top=True,
             input_tensor=None,
              pooling=None):

    # Determine proper input shape
    input_shape = (window_size,1)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    # Block 1
    x = Conv1D(k*64, (3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Conv1D(k*64, (3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling1D((2), strides=(2), name='block1_pool')(x)

    # Block 2
    x = Conv1D(k*128, (3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv1D(k*128, (3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling1D((2), strides=(2), name='block2_pool')(x)

    # Block 3
    x = Conv1D(k*256, (3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv1D(k*256, (3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv1D(k*256, (3), activation='relu', padding='same', name='block3_conv3')(x)
    x = Conv1D(k*256, (3), activation='relu', padding='same', name='block3_conv4')(x)
    x = MaxPooling1D((2), strides=(2), name='block3_pool')(x)

    # Block 4
    x = Conv1D(k*512, (3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv1D(k*512, (3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv1D(k*512, (3), activation='relu', padding='same', name='block4_conv3')(x)
    x = Conv1D(k*512, (3), activation='relu', padding='same', name='block4_conv4')(x)
    x = MaxPooling1D((2), strides=(2), name='block4_pool')(x)
    
    # Block 5
    x = Conv1DTranspose(k*512, (3), activation='relu', padding='same', name='block5_tran_conv1')(x)
    x = Conv1DTranspose(k*512, (3), activation='relu', padding='same', name='block5_tran_conv2')(x)
    x = Conv1DTranspose(k*512, (3), activation='relu', padding='same', name='block5_tran_conv3')(x)
    x = Conv1DTranspose(k*512, (3), activation='relu', padding='same', name='block5_tran_conv4')(x)
    x = AveragePooling1D((2), strides=(2), name='block5_pool')(x)
    
    # GRU layer, btw GRU stands for Gated Recurrent Units; https://en.wikipedia.org/wiki/Gated_recurrent_unit
    if method == 'gru':
        print("GRU enabled")
        x = GRU(method_num, activation='tanh', recurrent_activation='sigmoid')(x)
        
    if method == 'gru2':
        print("GRU enabled")
        x = GRU(method_num,activation='tanh',recurrent_activation='sigmoid',reset_after=True)(x)        
        
    if method == 'bigru':
        print("Bi-GRU enabled")
        x = Bidirectional(GRU(method_num, activation='tanh', recurrent_activation='sigmoid'))(x)
        
    if method == 'lstm':
        print("LSTM enabled")
        x = LSTM(method_num, activation='tanh', recurrent_activation='sigmoid')(x)
    
    # LSTM layer, btw LSTM stands for long short-term memory; https://en.wikipedia.org/wiki/Long_short-term_memory
    if include_top:
        # Classification block
        x = Flatten(name='flatten')(x)
        x = Dense(4096, activation='relu', name='fc1')(x)
        x = Dense(4096, activation='relu', name='fc2')(x)
        x = Dense(classes, activation='sigmoid', name='predictions')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling1D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling1D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='PC0')
    return model
##############################################################################################    
##############################################################################################
##############################################################################################
##############################################################################################    
##############################################################################################
##############################################################################################
##############################################################################################    
##############################################################################################
##############################################################################################
##############################################################################################    
##############################################################################################
##############################################################################################
##############################################################################################    
##############################################################################################
##############################################################################################
##############################################################################################    
##############################################################################################
##############################################################################################
def PC0_reg(classes,
             window_size,
             method,
             method_num,
             k,
             include_top=True,
             input_tensor=None,
              pooling=None,
             lambda_l2=0.001,
            ):

    # Determine proper input shape
    input_shape = (window_size,1)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    # Block 1
    x = Conv1D(k*64, (3), activation='relu', padding='same',kernel_regularizer=regularizers.l2(lambda_l2), name='block1_conv1')(img_input)
    x = Conv1D(k*64, (3), activation='relu', padding='same',kernel_regularizer=regularizers.l2(lambda_l2), name='block1_conv2')(x)
    x = MaxPooling1D((2), strides=(2), name='block1_pool')(x)

    # Block 2
    x = Conv1D(k*128, (3), activation='relu', padding='same',kernel_regularizer=regularizers.l2(lambda_l2),  name='block2_conv1')(x)
    x = Conv1D(k*128, (3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(lambda_l2), name='block2_conv2')(x)
    x = MaxPooling1D((2), strides=(2), name='block2_pool')(x)

    # Block 3
    x = Conv1D(k*256, (3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv1D(k*256, (3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv1D(k*256, (3), activation='relu', padding='same', name='block3_conv3')(x)
    x = Conv1D(k*256, (3), activation='relu', padding='same', name='block3_conv4')(x)
    x = MaxPooling1D((2), strides=(2), name='block3_pool')(x)

    # Block 4
    x = Conv1D(k*512, (3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv1D(k*512, (3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv1D(k*512, (3), activation='relu', padding='same', name='block4_conv3')(x)
    x = Conv1D(k*512, (3), activation='relu', padding='same', name='block4_conv4')(x)
    x = MaxPooling1D((2), strides=(2), name='block4_pool')(x)
    
    # Block 5
    x = Conv1DTranspose(k*512, (3), activation='relu', padding='same', name='block5_tran_conv1')(x)
    x = Conv1DTranspose(k*512, (3), activation='relu', padding='same', name='block5_tran_conv2')(x)
    x = Conv1DTranspose(k*512, (3), activation='relu', padding='same', name='block5_tran_conv3')(x)
    x = Conv1DTranspose(k*512, (3), activation='relu', padding='same', name='block5_tran_conv4')(x)
    x = AveragePooling1D((2), strides=(2), name='block5_pool')(x)
    
    # GRU layer, btw GRU stands for Gated Recurrent Units; https://en.wikipedia.org/wiki/Gated_recurrent_unit
    if method == 'gru':
        print("GRU enabled")
        x = GRU(method_num, activation='tanh', recurrent_activation='sigmoid')(x)
        
    if method == 'gru2':
        print("GRU enabled")
        x = GRU(method_num,activation='tanh',recurrent_activation='sigmoid',reset_after=True)(x)        
        
    if method == 'bigru':
        print("Bi-GRU enabled")
        x = Bidirectional(GRU(method_num, activation='tanh', recurrent_activation='sigmoid'))(x)
        
    if method == 'lstm':
        print("LSTM enabled")
        x = LSTM(method_num, activation='tanh', recurrent_activation='sigmoid')(x)
    
    # LSTM layer, btw LSTM stands for long short-term memory; https://en.wikipedia.org/wiki/Long_short-term_memory
    if include_top:
        # Classification block
        x = Flatten(name='flatten')(x)
        x = Dense(4096, activation='relu', name='fc1',kernel_regularizer=regularizers.l2(lambda_l2))(x)
        x = Dense(4096, activation='relu', name='fc2',kernel_regularizer=regularizers.l2(lambda_l2))(x)
        x = Dense(classes, activation='sigmoid', name='predictions')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling1D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling1D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='PC0')
    return model