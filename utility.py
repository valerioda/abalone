import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import quad
import scipy.special as sc
from scipy.integrate import odeint
import utility as psu
import scipy.integrate as integ
from scipy.optimize import curve_fit
import time
from matplotlib.colors import LogNorm
from numpy.random import randn
import matplotlib.colors as colors
import matplotlib.cbook as cbook
from matplotlib import cm


def read_file(filename , samples = 1024 ):
    data = np.fromfile(filename,  dtype=np.int16)
    n = len(data)/samples
    print(filename,'Number of events',n)
    data = np.array(np.array_split(np.array(data),n))
    return data

def plot_waveforms( data, events = 5 , tlim = None):
    print('n. events',data.shape[0],'waveform samples',data.shape[1])
    dts = np.arange(0,data.shape[1]/100,0.01)
    plt.figure(figsize=(12,6))
    for i in range(events):
        plt.plot(dts,data[i][:])
    if tlim is not None: plt.xlim(tlim[0],tlim[1])
    plt.xlabel('time ($\mu$s)',ha='right',x=1)
    plt.ylabel('channels',ha='right',y=1)

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
        
        
        
def process_abalone_data(data, nn=0,nplot=5):
    print(f'Total events number: {len(data)}')
    MINs, FOMs, MAXs, INTs, width50 = [], [], [], [], []
    ts = time.time()
    #plt.figure(figsize=(12,6))
    if nn == 0: nn = len(data)
    for i in range(nn):
        if (i % 200000) == 0: print(f'event n. {i} time to process: {time.time()-ts:.2f}')
        bl=np.max(data[i][:])
        wf = bl-data[i]
        #max_pos = np.where(wf==np.max(wf[512-50:512+200]))[0][0]
        max_pos = np.where(wf==np.max(wf[0:512+200]))[0][0]
        #datacut = data[np.array(max_pos)>1e6]
        area = np.sum(wf[max_pos-10:max_pos+90])
        #area = np.sum(wf)
        #charge = 1.602*1e-19
        #PE = (np.sum(wf))/charge
        fom = np.sum(wf[max_pos:])/area
        ii, areafrac = 1, 0
        while areafrac < 0.5:
            ilo50, ihi50 = max_pos-ii, max_pos+ii
            areafrac = np.sum(wf[ilo50:ihi50])/area
            ii += 1
            wid50 = ihi50 - ilo50
        MINs.append(bl)
        MAXs.append(np.max(wf))
        FOMs.append(fom)
        INTs.append(area)
        width50.append(wid50)
        #if i < nplot: plt.plot(wf,label=f'A={area}, f.o.m.={fom:.2f}')
    #plt.legend()
    return np.array(MINs), np.array(FOMs), np.array(MAXs), np.array(INTs), np.array(width50)

def plot_area_max1(AREAs,MAXs,bins=200):
    area_space = np.logspace(1.5,7.5,bins)
    max_space = np.logspace(1,4.5,bins)
    plt.figure(figsize=(12,6))
    #fig, axs = plt.subplots(1, 1, figsize=(18,5))
    axs = plt.hist2d(AREAs,MAXs,bins=(area_space,max_space),norm=matplotlib.colors.LogNorm())
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Area', ha='right', y=1, fontsize=12)
    plt.ylabel('MAXpeaks', ha='right', y=1, fontsize=12)
    #plt.title()
    plt.colorbar()
    #pcm = axs[0].pcolormesh(AREAs, MAXs, bins=(area_space,max_space), norm=matplotlib.colors.LogNorm())
    #cmp=fig.colorbar(axs[4])
    #axs[1].hist2d(AREAs_1,MAXs_1,bins=(area_space,max_space),norm=matplotlib.colors.LogNorm())
    #axs[1].set_xscale('log')
    #axs[1].set_yscale('log')
    #axs[1].set_xlabel('Area', ha='right', y=1, fontsize=12)
    #axs[1].set_ylabel('MAXpeaks', ha='right', y=1, fontsize=12)
    #axs[1].set_title('ABALONE 14k 10_05_2021_LED')
    #axs[1].set_title(title[1])
    #cmaps = ['RdBu_r', 'viridis']
    #for col in range(2):
     #   for row in range(1):
      #      ax = (axs0,axs1)[1,2]
       #     pcm = ax.pcolormesh(np.random.random((20, 20)) * (col + 1),
        #                    cmap=cmaps[col])
    #fig.colorbar(pcm, ax=(axs0, axs1)[:, col], shrink=0.6)
    #axs[2].hist2d(AREAs_2,MAXs_2,bins=(area_space,max_space),norm=matplotlib.colors.LogNorm())
    #axs[2].set_xscale('log')
    #axs[2].set_yscale('log')
    #axs[2].set_xlabel('area', ha='right', y=1, fontsize=12)
    #axs[2].set_ylabel('MAXpeaks', ha='right', y=1, fontsize=12)
    #axs[2].set_title(title[2])
    

def plot_area_max2(AREAs,MAXs,AREAs_1,MAXs_1,bins=200):
    area_space = np.logspace(2,7.0,bins)
    max_space = np.logspace(1,4.5,bins)
    fig, axs = plt.subplots(1, 2, figsize=(18,5))
    axs[0].hist2d(AREAs,MAXs,bins=(area_space,max_space),norm=matplotlib.colors.LogNorm())
    axs[0].set_xscale('log')
    axs[0].set_yscale('log')
    axs[0].set_xlabel('Area', ha='right', y=1, fontsize=12)
    axs[0].set_ylabel('MAXpeaks', ha='right', y=1, fontsize=12)
    #axs[0].set_title(title[0])
    #pcm = axs[0].pcolormesh(AREAs, MAXs, bins=(area_space,max_space), norm=matplotlib.colors.LogNorm())
    #cmp=fig.colorbar(axs[4])
    axs[1].hist2d(AREAs_1,MAXs_1,bins=(area_space,max_space),norm=matplotlib.colors.LogNorm())
    axs[1].set_xscale('log')
    axs[1].set_yscale('log')
    axs[1].set_xlabel('Area', ha='right', y=1, fontsize=12)
    axs[1].set_ylabel('MAXpeaks', ha='right', y=1, fontsize=12)
    #axs[1].set_title('ABALONE 14k 10_05_2021_LED')
    #axs[1].set_title(title[1])
    #cmaps = ['RdBu_r', 'viridis']
    #for col in range(2):
     #   for row in range(1):
      #      ax = (axs0,axs1)[1,2]
       #     pcm = ax.pcolormesh(np.random.random((20, 20)) * (col + 1),
        #                    cmap=cmaps[col])
    #fig.colorbar(pcm, ax=(axs0, axs1)[:, col], shrink=0.6)
    #axs[2].hist2d(AREAs_2,MAXs_2,bins=(area_space,max_space),norm=matplotlib.colors.LogNorm())
    #axs[2].set_xscale('log')
    #axs[2].set_yscale('log')
    #axs[2].set_xlabel('area', ha='right', y=1, fontsize=12)
    #axs[2].set_ylabel('MAXpeaks', ha='right', y=1, fontsize=12)
    #axs[2].set_title(title[2])
    
    
    
def plot_area_max3(AREAs,MAXs,AREAs_1,MAXs_1,AREAs_2,MAXs_2,title,bins=200):
    area_space = np.logspace(4,6.5,bins)
    max_space = np.logspace(1,4.5,bins)
    fig, axs = plt.subplots(1, 3, figsize=(18,5))
    axs[0].hist2d(AREAs,MAXs,bins=(area_space,max_space),norm=matplotlib.colors.LogNorm())
    axs[0].set_xscale('log')
    axs[0].set_yscale('log')
    axs[0].set_xlabel('PE', ha='right', y=1, fontsize=12)
    axs[0].set_ylabel('MAXpeaks', ha='right', y=1, fontsize=12)
    axs[0].set_title(title[0])
    #pcm = axs[0].pcolormesh(AREAs, MAXs, bins=(area_space,max_space), norm=matplotlib.colors.LogNorm())
    #cmp=fig.colorbar(axs[4])
    axs[1].hist2d(AREAs_1,MAXs_1,bins=(area_space,max_space),norm=matplotlib.colors.LogNorm())
    axs[1].set_xscale('log')
    axs[1].set_yscale('log')
    axs[1].set_xlabel('PE', ha='right', y=1, fontsize=12)
    axs[1].set_ylabel('MAXpeaks', ha='right', y=1, fontsize=12)
    #axs[1].set_title('ABALONE 14k 10_05_2021_LED')
    axs[1].set_title(title[1])
    #cmaps = ['RdBu_r', 'viridis']
    #for col in range(2):
     #   for row in range(1):
      #      ax = (axs0,axs1)[1,2]
       #     pcm = ax.pcolormesh(np.random.random((20, 20)) * (col + 1),
        #                    cmap=cmaps[col])
    #fig.colorbar(pcm, ax=(axs0, axs1)[:, col], shrink=0.6)
    axs[2].hist2d(AREAs_2,MAXs_2,bins=(area_space,max_space),norm=matplotlib.colors.LogNorm())
    axs[2].set_xscale('log')
    axs[2].set_yscale('log')
    axs[2].set_xlabel('area', ha='right', y=1, fontsize=12)
    axs[2].set_ylabel('MAXpeaks', ha='right', y=1, fontsize=12)
    axs[2].set_title(title[2])
    
    
def plot_area_fom1(AREAs,FOMs,bins=200):
    area_space = np.logspace(4,6.5, bins)
    plt.figure(figsize=(12,6))
    #fig, axs = plt.subplots(1, 1, figsize=(18,6))
    axs = plt.hist2d(AREAs,FOMs,bins=(area_space,bins),norm=matplotlib.colors.LogNorm())
    plt.xscale('log')
    plt.xlabel('area',ha='right',x=1,fontsize=12)
    plt.ylabel('FOM',ha='right',y=1,fontsize=12)
    #plt.title(title)
    #axs[0].set_title('ABALONE 14k 10_05_2021_noLED')
    #axs[1].hist2d(AREAs_1,FOMs_1,bins=(area_space,bins),norm=matplotlib.colors.LogNorm())
    #axs[1].set_xscale('log')
    #axs[1].set_xlabel('area',ha='right',x=1,fontsize=12)
    #axs[1].set_ylabel('FOM',ha='right',y=1,fontsize=12)
    #axs[1].set_title(title[1])
    #axs[1].set_title('ABALONE 14k 10_05_2021_LED')
    #axs[2].hist2d(AREAs_2,FOMs_2,bins=(area_space,bins),norm=matplotlib.colors.LogNorm())
    #axs[2].set_xscale('log')
    #axs[2].set_xlabel('area', ha='right', x=1, fontsize=12)
    #axs[2].set_ylabel('FOM', ha='right', y=1, fontsize=12)
    #axs[2].set_title(title[2])

    
    
def plot_area_fom2(AREAs,FOMs,AREAs_1,FOMs_1,bins=200):
    area_space = np.logspace(4,6.5, bins)
    fig, axs = plt.subplots(1, 2, figsize=(18,6))
    axs[0].hist2d(AREAs,FOMs,bins=(area_space,bins),norm=matplotlib.colors.LogNorm())
    axs[0].set_xscale('log')
    axs[0].set_xlabel('area',ha='right',x=1,fontsize=12)
    axs[0].set_ylabel('FOM',ha='right',y=1,fontsize=12)
    #axs[0].set_title(title[0])
    #axs[0].set_title('ABALONE 14k 10_05_2021_noLED')
    axs[1].hist2d(AREAs_1,FOMs_1,bins=(area_space,bins),norm=matplotlib.colors.LogNorm())
    axs[1].set_xscale('log')
    axs[1].set_xlabel('area',ha='right',x=1,fontsize=12)
    axs[1].set_ylabel('FOM',ha='right',y=1,fontsize=12)
    #axs[1].set_title(title[1])
    #axs[1].set_title('ABALONE 14k 10_05_2021_LED')
    #axs[2].hist2d(AREAs_2,FOMs_2,bins=(area_space,bins),norm=matplotlib.colors.LogNorm())
    #axs[2].set_xscale('log')
    #axs[2].set_xlabel('area', ha='right', x=1, fontsize=12)
    #axs[2].set_ylabel('FOM', ha='right', y=1, fontsize=12)
    #axs[2].set_title(title[2])
    
    
def plot_area_fom3(AREAs,FOMs,AREAs_1,FOMs_1,AREAs_2,FOMs_2,title,bins=200):
    area_space = np.logspace(4,6.5, bins)
    fig, axs = plt.subplots(1, 3, figsize=(18,6))
    axs[0].hist2d(AREAs,FOMs,bins=(area_space,bins),norm=matplotlib.colors.LogNorm())
    axs[0].set_xscale('log')
    axs[0].set_xlabel('area',ha='right',x=1,fontsize=12)
    axs[0].set_ylabel('FOM',ha='right',y=1,fontsize=12)
    axs[0].set_title(title[0])
    #axs[0].set_title('ABALONE 14k 10_05_2021_noLED')
    axs[0].set_title(title[0])
    axs[1].hist2d(AREAs_1,FOMs_1,bins=(area_space,bins),norm=matplotlib.colors.LogNorm())
    axs[1].set_xscale('log')
    axs[1].set_xlabel('area',ha='right',x=1,fontsize=12)
    axs[1].set_ylabel('FOM',ha='right',y=1,fontsize=12)
    axs[1].set_title(title[1])
    #axs[1].set_title('ABALONE 14k 10_05_2021_LED')
    axs[2].hist2d(AREAs_2,FOMs_2,bins=(area_space,bins),norm=matplotlib.colors.LogNorm())
    axs[2].set_xscale('log')
    axs[2].set_xlabel('area', ha='right', x=1, fontsize=12)
    axs[2].set_ylabel('FOM', ha='right', y=1, fontsize=12)
    axs[2].set_title(title[2])

    

def plot_area_width2(AREAs,width50,AREAs_1,width50_1,bins=200):
    area_space = np.logspace(2,7, bins)
    width_space = np.logspace(1,3.5, bins)
    fig, axs = plt.subplots(1,2, figsize=(14,5))
    axs[0].hist2d(AREAs,width50,bins=(area_space,width_space),norm=matplotlib.colors.LogNorm())
    axs[0].set_xscale('log')
    axs[0].set_yscale('log')
    axs[0].set_xlabel('area',ha='right',x=1,fontsize=12)
    axs[0].set_ylabel('width 50%',ha='right',y=1,fontsize=12)
    #axs[0].set_title([0])
    axs[1].hist2d(AREAs_1,width50_1,bins=(area_space,width_space),norm=matplotlib.colors.LogNorm())
    axs[1].set_xscale('log')
    axs[1].set_yscale('log')
    axs[1].set_xlabel('area',ha='right',x=1,fontsize=12)
    axs[1].set_ylabel('width 50%',ha='right',y=1,fontsize=12)
    #axs[1].set_title([1])
    #axs[2].hist2d(AREAs_2,width50_2,bins=(area_space,width_space),norm=matplotlib.colors.LogNorm(),cmap=plt.cm.jet)
    #axs[2].set_xscale('log')
    #axs[2].set_yscale('log')
    #axs[2].set_xlabel('area', ha='right', y=1, fontsize=12)
    #axs[2].set_ylabel('width 50%', ha='right', y=1, fontsize=12)
    #axs[2].set_title('ABALONE 10k 07_05_2021_LEDthreshold') 

def plot_area_width1(AREAs,width50,bins=200):
    area_space = np.logspace(1.5,7, bins)
    width_space = np.logspace(0,3.5, bins)
    plt.figure(figsize=(12,6))
    #fig, axs = plt.subplots(1,3, figsize=(14,5))
    axs = plt.hist2d(AREAs,width50,bins=(area_space,width_space),norm=matplotlib.colors.LogNorm())
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('area',ha='right',x=1,fontsize=12)
    plt.ylabel('width 50%',ha='right',y=1,fontsize=12)
    #plt.title()
    plt.colorbar()
   # axs[1].hist2d(AREAs_1,width50_1,bins=(area_space,width_space),norm=matplotlib.colors.LogNorm(),cmap=plt.cm.jet)   
    
    
    
def data_analysisNoLED(filename='/data/abalone/2021_05_10/SiPM2_32V_ABALONE_14k_NoLED.dat', volts=14):
    data = read_file(filename)
    MINs, FOMs, MAXs, INTs = process_abalone_data(data, nn=1000, nplot=10)
    Area_MAX = plot_area_max(INTs,MAXs,INTs_1,MAXs_1,bins=200, volts=volts)
    Area_FOM = plot_area_fom(INTs,MAXs,INTs_1,MAXs_1,bins=200, volts=volts)

def data_analysisLED(filename='/data/abalone/2021_05_10/SiPM2_32V_ABALONE_14k_LED4V.dat', volts=14):
    data = read_file(filename)
    MINs, FOMs, MAXs, INTs = process_abalone_data(data, nn=1000, nplot=10)
    Area_MAX = plot_area_max(INTs_2,MAXs_2,INTs_3,MAXs_3,bins=200, volts=volts)
    Area_FOM = plot_area_fom(INTs_2,MAXs_2,INTs_3,MAXs_3,bins=200, volts=volts)

    
    
########### NEW ANALYSIS FUNCTION ##################

def plot_area_spectrum(AREAs,bins=200, logx=True, logy=True,volts=10,low=3.5, high=6):
    area_space = np.logspace(low,high, bins)
    plt.figure(figsize=(12,6))
    a=plt.hist(AREAs,bins=area_space,histtype='step',lw=2,density=False)
    if logx==True: plt.xscale('log')
    if logy==True: plt.yscale('log')
    plt.title(f'ABALONE at {volts} kV')
    plt.xlabel('area (ADC x $\mu$s)',ha='right',x=1)
    plt.ylabel('counts',ha='right',y=1)
    
    
def plot_area_max(AREAs,MAXs,bins=200,volts=10,low=3.5, high=6,low2=0.5, high2=4.5):
    area_space = np.logspace(low,high,bins)
    max_space = np.logspace(low2,high2,bins)
    plt.figure(figsize=(12,6))
    fig, axs = plt.subplots(1, 1, figsize=(12,6))
    axs = plt.hist2d(AREAs,MAXs,bins=(area_space,max_space),norm=matplotlib.colors.LogNorm())
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('area (ADC x $\mu$s)', ha='right', x=1, fontsize=12)
    plt.ylabel('MAXpeaks', ha='right', y=1, fontsize=12)
    plt.title(f'ABALONE at {volts} kV')
    plt.colorbar()
    
def plot_area_fom(AREAs,FOMs,bins=200,volts=10,low=3.5, high=6, low_fom = 0, high_fom = 5):
    area_space = np.logspace(low,high,bins)
    fom_space = np.linspace(low_fom,high_fom,bins)
    plt.figure(figsize=(12,6))
    fig, axs = plt.subplots(1, 1, figsize=(12,6))
    axs = plt.hist2d(AREAs,FOMs,bins=(area_space,fom_space),norm=matplotlib.colors.LogNorm())
    plt.xscale('log')
    #plt.ylim(-10,10)
    #plt.xlim(0,1e6)
    plt.xlabel('area (ADC x $\mu$s)',ha='right',x=1,fontsize=12)
    plt.ylabel('FOM',ha='right',y=1,fontsize=12)
    plt.title(f'ABALONE at {volts} kV')
    plt.colorbar()
    
def plot_area_width(AREAs,width50,bins=200,volts=10,low=3.5, high=6,low2=1.5, high2=4.5):
    area_space = np.logspace(low,high, bins)
    width_space = np.logspace(low2,high2, bins)
    plt.figure(figsize=(12,6))
    fig, axs = plt.subplots(1, 1, figsize=(12,6))
    axs = plt.hist2d(AREAs,width50,bins=(area_space,width_space),norm=matplotlib.colors.LogNorm())
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('area (ADC x $\mu$s)',ha='right',x=1,fontsize=12)
    plt.ylabel('width 50%',ha='right',y=1,fontsize=12)
    plt.title(f'ABALONE at {volts} kV')
    plt.colorbar()

        
def plot_area_entropy(AREAs,ENTRs,bins=200,volts=10,low=3.5, high=6,low2=0, high2=3.5):
    area_space = np.logspace(low, high, bins)
    entr_space = np.linspace(low2,high2, bins)
    plt.figure(figsize=(12,6))
    a=plt.hist2d(AREAs,ENTRs,bins=(area_space,entr_space),norm=matplotlib.colors.LogNorm())
    plt.xscale('log')
    plt.xlabel('area (ADC x $\mu$s)',ha='right',x=1,fontsize=12)
    plt.ylabel('entropy',ha='right',y=1,fontsize=12)
    plt.title(f'ABALONE at {volts} kV')
    plt.colorbar()
    
def plot_area_risetime(AREAs,RTs,bins=200,volts=10,low=3.5, high=6,low2=-1, high2=450):
    area_space = np.logspace(low, high, bins)
    rt_space = np.linspace(low2,high2, bins)
    plt.figure(figsize=(12,6))
    a = plt.hist2d(AREAs,RTs,bins=(area_space,rt_space),norm=matplotlib.colors.LogNorm())
    plt.xscale('log')
    #plt.yscale('log')
    plt.xlabel('area (ADC x $\mu$s)',ha='right',x=1,fontsize=12)
    plt.ylabel('risetime (ns)',ha='right',y=1,fontsize=12)
    plt.title(f'ABALONE at {volts} kV')
    plt.colorbar()
    
def plot_risetime_entropy(RTs,ENTRs,bins=200,volts=10,low=0, high=450,low2=1, high2=200):
    rt_space = np.linspace(low,high, bins)
    entr_space = np.linspace(low2,high2, bins)
    plt.figure(figsize=(12,6))
    a = plt.hist2d(RTs,ENTRs,bins=(rt_space,entr_space),norm=matplotlib.colors.LogNorm())
    #plt.xscale('log')
    plt.xlabel('risetime (ns)',ha='right',x=1,fontsize=12)
    plt.ylabel('entropy',ha='right',y=1,fontsize=12)
    plt.title(f'ABALONE at {volts} kV')
    plt.colorbar()
    
def plot_area_tau(AREAs,TAUs,bins=200,volts=10,low=3.5, high=6,low2=0.5, high2=4.5):
    area_space = np.logspace(low,high,bins)
    tau_space = np.linspace(low2,high2,bins)
    plt.figure(figsize=(12,6))
    fig, axs = plt.subplots(1, 1, figsize=(12,6))
    axs = plt.hist2d(AREAs,TAUs,bins=(area_space,tau_space),norm=matplotlib.colors.LogNorm())
    plt.xscale('log')
    #plt.yscale('log')
    plt.xlabel('area (ADC x $\mu$s)', ha='right', x=1, fontsize=12)
    plt.ylabel('tau', ha='right', y=1, fontsize=12)
    plt.title(f'ABALONE at {volts} kV')
    plt.colorbar()
