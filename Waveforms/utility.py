import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.integrate as integ
from scipy.optimize import curve_fit
from scipy.signal import find_peaks


def read_fileLED( filename, samples = 1024 ):
    dataLED = np.fromfile(filename,  dtype=np.int16)
    n = len(dataLED)/samples
    print(filename,'Number of events',n)
    dataLED = np.array(np.array_split(np.array(dataLED),n))
    return dataLED

def read_filenoLED( filename, samples = 1024 ):
    datanoLED = np.fromfile(filename,  dtype=np.int16)
    n = len(datanoLED)/samples
    print(filename,'Number of events',n)
    datanoLED = np.array(np.array_split(np.array(datanoLED),n))
    return datanoLED


def plot_waveformsnoLED( datanoLED, events = 5 ):
    dts = np.arange(0,datanoLED.shape[1]*10,10)
    fig, axs = plt.subplots(events, figsize=(12,10))
    for i in range(events):
        axs[i].plot(dts,datanoLED[i][:])
        axs[i].set(xlabel='time (ns)', ylabel='channels')
        
        
def plot_waveformsLED( dataLED, events = 5 ):
    dts = np.arange(0,dataLED.shape[1]*10,10)
    fig, axs = plt.subplots(events, figsize=(12,10))
    for i in range(events):
        axs[i].plot(dts,dataLED[i][:])
        axs[i].set(xlabel='time (ns)', ylabel='channels')
        
        
