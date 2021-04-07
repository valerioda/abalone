import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import quad
import scipy.special as sc
from scipy.integrate import odeint
import scipy.integrate as integ
from scipy.optimize import curve_fit
import time
import json
import os

def read_waveforms_from_json( PE = 1, angle = 0, plot = False ):
    wfs = []
    path = '/home/dandrea/abalone_simulation/results/SiPM'
    t_start = time.time()
    #os.listdir(directory)
    i = 1
    filename = f'{path}/SiPM_readout_{PE}_{angle}_run_{i}.txt'
    while os.path.isfile(filename):
        with open(filename) as f:
            for jsonObj in f:
                wfDict = json.loads(jsonObj)
                try: wfs.append(wfDict['voltages'])
                except: pass
        filename = f'{path}/SiPM_readout_{PE}_{angle}_run_{i}.txt'
        i += 1
    nwf = len(wfs)
    print(f'Number of waveforms {nwf}, time to read {time.time() - t_start:.2f}')
    if plot:
        plt.figure(figsize=(8,4.5))
        for i in range(10):
            plt.plot(wfs[i],label=f'waveform {i}')
        plt.xlabel('time (ns)', ha='right', x=1)
        plt.ylabel('voltages (a.u.)', ha='right', y=1)
        #plt.ylim(0,2)
        #plt.legend()
        plt.savefig(f'waveforms_PE{PE}_angle{angle}.png',dpi=600)
    return np.array(wfs)


def search_peaks(wf, n, ampllim = 9, plot = False): # search all the peaks of the signal
    # n: nb of bin-size for the dled delay (dled = "discret derivative")
    # ampllim: min amplitude to select a peak
    L = []
    t = np.arange(0,len(wf))
    # derivative
    dled = wf[n:] - wf[:-n]
    if plot:
        plt.figure(figsize=(8,4.5))
        plt.plot(t[:-n], dled, label = 'dled signal')
        plt.xlabel(r'samples',ha='right',x=1)
        plt.ylabel('amplitude',ha='right',y=1)
        plt.legend()
    N = np.array([i for i in range(len(dled))]) # we will work in term of bin-size instead of time
    N1 = N[dled >= ampllim]
    while len(N1) != 0:
        n1 = N1[0]
        N, dled = N[N >= n1], dled[N >= n1]
        N2 = N[dled < ampllim]
        if len(N2) != 0:
            n2 = N2[0]
        else:
            n2 = N[-1]
        L.append(n1)
        L.append(n2)
        N, dled = N[N > n2], dled[N > n2]
        N1 = N[dled >= ampllim]
        if plot: plt.plot( (n1+n2)/2, 0, "x")
    #L.append(len(wf)-1)
    return L


def expo_neg(c):
    return(lambda x, a, b : c - a*np.exp(-b*x))


def expo_pos(c):
    return(lambda x, a, b : c + a*np.exp(-b*x))


def integral_simulation_peaks( wf, peaks_list, dtl = -2, dtr = 1,
                  tfit = 20, tlim = 100, tc = 5, tll = 5, tlr = 10, central=True, plot=False):
    # tlim = time window of the fit integration
    # tfit = time window of the fit
    # tc = min time to consider 2 peaks independently
    # tll = left limit of the window to search baseline value
    # tlr = right limit of the window to search Amax
    inttot = 0
    tt = np.array([i for i in range(len(wf))]) # time in bin-size
    tplot_tot = []
    A_tot = []
    integrals = []
    for i in range(len(peaks_list)//2):
        dt = peaks_list[2*i + 1] - peaks_list[2*i]
        if dt < tc: continue #skip peaks if is too close to next one
        tlo = peaks_list[2*i]-tll
        if tlo < 0: tlo = 0
        bl = np.max(wf[tlo:peaks_list[2*i]+1])
        Am = wf[peaks_list[2*i]:peaks_list[2*i+1] + tlr]
        ttm = tt[peaks_list[2*i]:peaks_list[2*i+1] + tlr]
        amp = np.max(Am) #local amplitude of the signal
        tmin = ttm[Am == amp][0]
        tl = tt[(tt <= tmin+dtr) & (tt >= tmin+dtl)]
        wfl = wf[(tt <= tmin+dtr) & (tt >= tmin+dtl)]
        Il = integ.simps(wfl, tl)
        
        # calculation of real integral of the waveform
        real_t = tt[(tt <= tmin+tlim) & (tt >= tmin+dtl)]
        real_wf = wf[(tt <= tmin+tlim) & (tt >= tmin+dtl)]
        intreal = integ.simps(real_wf, real_t)
        tr = tt[tmin+dtr:tmin+tfit] # time window for the fit
        if central: mask = (len(tr) >= 5 and amp > 5 and np.abs(tmin-len(wf)/2) < 50)
        else: mask = (len(tr) >= 5 and amp < 5 and np.abs(tmin-len(wf)/2) < 900)
        if mask:
            tr2 = tr - tr[0]
            #print('tmin:',tmin,'bl:',bl,'amp max:',amp)
            #try:
            fct_fit = expo_pos(bl) # fct used for the fit
            popt, pcov = curve_fit(fct_fit, tr2, wf[tr],
                                    p0 = np.array([amp, 5/100]),
                                    bounds =  ([amp/1.5, 1/100], [amp*1.5, 10/100]))
            a, b = popt
            tnew = tt[tr[0]:]
            if plot:
                plt.figure(figsize=(8,4.5))
                fct_fit_tot = fct_fit((tnew-tr[0]),a,b)
                tnew2 = min(tlim, tt[-1]-tr[0])
                tplot = tt[tmin+dtl-20:tmin+tnew2+20]
                plt.plot(tplot,wf[tmin+dtl-20:tmin+tnew2+20],label='SiPM signal')
                plt.plot(tnew[:tnew2], fct_fit_tot[:tnew2],
                         label=f'fit f(x) = baseline - a*exp(-b*x):\n a = {a:.2f}, b = {b:.2f}')
                plt.axhline(bl, color = 'r', label = 'baseline')
                plt.vlines((tmin+dtl), wf[tmin+dtl]-10, bl+10, colors = 'g',
                           label = 'integration limits')
                plt.vlines((tmin+dtr), wf[tmin+dtr]-10, bl+10, colors = 'g')
                plt.vlines((tmin+tfit), wf[tmin+tfit]-10, bl+10, colors = 'c',
                           label = 'fit limit')
                plt.vlines((tmin+tlim), wf[tmin+tlim]-10, bl+10, colors = 'g')
                #plt.xlabel(r'time ($\mu s$)',ha='right',x=1)
                #plt.ylabel('amplitude',ha='right',y=1)
                plt.legend()
                tlimplot = tr[0]
            fct_int = lambda x : bl + fct_fit(x, a, b)
            #print('bl, a, b : ', bl, a, b)
            Ir, err = integ.quad(fct_int, 0, (tlim-dtr))
            Ir, err = integ.quad(fct_int, 0, (tlim-dtr)/100)
            inttot = Il + Ir
            integrals.append(inttot)
            #print(f'Amp: {amp:.2f} Integral: {Il:.3f} + {Ir:.3f} = {inttot:.3f}')
            tl = tt[tt >= peaks_list[2*i+1]+dtl]
    return integrals