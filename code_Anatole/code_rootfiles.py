#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 14:15:13 2019

@author: anatole
"""










# you should totally read the other code before this one (to understand well) because there are more comments










import uproot
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integ
from scipy.optimize import curve_fit



#files from Abalone 25 kV

rfileA25L1 = '/media/anatole/MAXTOR/ABALONE/ABALONE_25kV_May/Abalone_LED_2019_05_21/raw/25kV_LED1.66V_SiPM30.9V_ext_LED_trigger/25kV_LED1.66V_SiPM30.9V_ext_LED_trigger_Module_0_0.root'
# LED with SiPM voltage = 30,9 V ; LED voltage can be 1.62, 1.64 or 1.66 V (à changer à 2 endroits !)

rfileA25L2 = '/media/anatole/MAXTOR/ABALONE/ABALONE_25kV_May/Abalone_LED_2019_05_10/raw/25kV_LED1.60V_SiPM32.9V_ext_trigger/25kV_LED1.60V_SiPM32.9V_ext_trigger_Module_0_0.root'
# LED with SiPM voltage = 32,9 V ; LED voltage can be 1.60, 1.62, 1.64 or 1.66 V (à changer à 2 endroits !)

rfileA25la = '/media/anatole/MAXTOR/ABALONE/ABALONE_25kV_May/Abalone_Laser_2019_05_10/raw/25kV_Laser1k15_SiPM32.9V_laser_self_trigger_only_500k_events/25kV_Laser1k15_SiPM32.9V_laser_self_trigger_only_500k_events_Module_0_0.root'
rfileA25lb = '/media/anatole/MAXTOR/ABALONE/ABALONE_25kV_May/Abalone_Laser_2019_05_10/raw/25kV_Laser10k15_SiPM32.9V_laser_self_trigger_only_5M_events/25kV_Laser10k15_SiPM32.9V_laser_self_trigger_only_5M_events_Module_0_0.root'
rfileA25lc = '/media/anatole/MAXTOR/ABALONE/ABALONE_25kV_May/Abalone_Laser_2019_05_10/raw/25kV_Laser100k15_SiPM32.9V_abalone_self_trigger_only_5M_events/25kV_Laser100k15_SiPM32.9V_abalone_self_trigger_only_5M_events_Module_0_0.root'
# laser


# files from SiPM only test :

rfileSL1 = '/media/anatole/MAXTOR/ABALONE/SiPM_june/LED_1.58V_19_06_07/SiPM3_34.0V_LED_1.58V/SiPM3_34.0V_LED_1.58V_Module_0_0.root'
# with LED at 1.58 V and SiPM at 30 or 31 or 32 or 33 or 34 V

rfileSL2 = '/media/anatole/MAXTOR/ABALONE/SiPM_june/LED_1.60V_19_06_07/SiPM3_34.0V_LED_1.60V/SiPM3_34.0V_LED_1.60V_Module_0_0.root'
# with LED at 1.60 V and SiPM at 30 or 31 or 32 or 33 or 34 V

rfileSl = '/media/anatole/MAXTOR/ABALONE/SiPM_june/laser_19_06_07/SiPM3_32.0V_laser/SiPM3_32.0V_laser_Module_0_0.root'
# with laser and SiPM at 30 or 31 or 32 or 33 or 34 V

rfileSl2 = '/media/anatole/MAXTOR/ABALONE/SiPM_june/laser_19_06_12/SiPM3_30.0V_laser/SiPM3_30.0V_laser_Module_0_0.root'
# with laser and SiPM at 30 or 31 or 32 or 33 or 34 V

rfileSl3 = '/media/anatole/MAXTOR/ABALONE/SiPM_june/laser_19_06_12/SiPM3_32.0V_laser10/SiPM3_32.0V_laser10_Module_0_0.root'
# with laser at 10 or 11 or 12 or 13 or 14 and SiPM at 32

rfileSl4a = '/media/anatole/MAXTOR/ABALONE/SiPM_june/laser_19_06_14/SiPM3_32.0V_laser9/SiPM3_32.0V_laser9_Module_0_0.root'
rfileSl4b = '/media/anatole/MAXTOR/ABALONE/SiPM_june/laser_19_06_14/SiPM3_32.0V_laser9.5/SiPM3_32.0V_laser9.5_Module_0_0.root'
rfileSl4c = '/media/anatole/MAXTOR/ABALONE/SiPM_june/laser_19_06_14/SiPM3_30.0V_laser9.5/SiPM3_30.0V_laser9.5_Module_0_0.root'
# with laser at 9 and SiPM at 32, or laser at 9,5 and SiPM at 30 or 32

def rfileSl5(m, n): # 5M events files
    n in [1, 5]
    if m == 1:
        return('/media/anatole/MAXTOR/ABALONE/SiPM_june/laser_19_06_27/SiPM3_32.9V_laser9.5/SiPM3_32.9V_laser9.5_Module_0_%s.root'%n)
    if m == 2:
        return('/media/anatole/MAXTOR/ABALONE/SiPM_june/laser_19_06_27/SiPM3_32.0V_laser9.5/SiPM3_32.0V_laser9.5_Module_0_%s.root'%n)
    if m == 3:
        return('/media/anatole/MAXTOR/ABALONE/SiPM_june/laser_19_06_27/SiPM3_32.0V_laser10/SiPM3_32.0V_laser10_Module_0_%s.root'%n)


# 2nd SiPM test

def rfileS2l(V_SiPM, P_laser, n): # V_SiPM at 26 or 26.5 & laser at 9.5, 10, 10.5
    return('/media/anatole/MAXTOR/ABALONE/SiPM_5.1/laser_19_07_01/SiPM5.1_{0}V_laser{1}/SiPM5.1_{0}V_laser{1}_Module_0_{2}.root'.format(float(V_SiPM), P_laser, n))


# Abalone 22kV :

def rfileA22L1(st): # st = '1.60', '1.65', '1.70', 1.75'
    return('/media/anatole/MAXTOR/ABALONE/ABALONE_22kV_March/LED_March_14/22kV_LED'+st+'V_SiPM32.9V/22kV_LED'+st+'V_SiPM32.9V_Module_0_0.root')

def rfileA22L2(V_SiPM, V_LED): # V_SiPM = 31, 32 or 32.9 ; V_LED = 1.6 or 1.65
    return('/media/anatole/MAXTOR/ABALONE/ABALONE_22kV_March/LED_March_19/22kV_LED{1}V_SiPM{0}V/22kV_LED{1}V_SiPM{0}V_Module_0_0.root'.format(float(V_SiPM), V_LED))


current_file = rfileSL1


def load(n): #load the nth event of the current file
    L = uproot.open(current_file)['t1']
    L2 = L.array('wf1')[n] # use 'wf0' for one specific file (I forgot which one but if this line raise an error it is probably that)
    t = np.linspace(0, (len(L2)-1)*10**-8, len(L2))
    return(t, L2)

def loadfile(): # load all the events of the current file
    L = uproot.open(current_file)['t1']
    L2 = L.array('wf1')
    t = np.linspace(0, (len(L2[0])-1)*10**-8, len(L2[0]))
    return(t, L2)



def plotI(n):
    t, I2 = load(n)
    plt.plot(t*10**6, I2, label = r'response signal')
    plt.xlabel(r'time ($\mu s$)')
    plt.ylabel('amplitude')
    plt.legend()



def expo(x, a, b):
    return(-a*np.exp(-b*x))

def integral(t, I, limits = False):
    t = np.array(t)*1e6
    I = np.array(I)
    bl = 12350 # base line #12345 for L1, L2 and l, 10000 for l2, l3
    I = I - bl
    #find min
    Tm = 1.37 # average minimum time
    dt = 0.1 # average deviation of the min
    t1, I1 = t[t > Tm - dt], I[t > Tm - dt]
    t2, I2 = t1[t1 < Tm + dt], I1[t1 < Tm + dt]
    tm = t2[I2 == np.min(I2)][0]
    t -= tm
    #cst direct integration
    dtl = -0.03 #cst left time limit for direct integration
    dtr = 0#.02 #cst right time limit of direct integration
    tl, Il = t[t <= dtr], I[t <= dtr]
    #tl, Il = t[t <= 0], I[t <= 0]
    intl = -1 * integ.simps(Il[tl >= dtl], tl[tl >= dtl])
    #fit and integration
    tr, Ir = t[t >= dtr], I[t >= dtr]
    #tr, Ir = t[t >= 0], I[t >= 0]
    tfit = 0.20 # cst time window for fit
    try:
        popt, pcov = curve_fit(expo, tr[tr < tfit], Ir[tr < tfit], p0 = np.array([200, 1]))
        a, b = popt
        tlim = 1 # time limit for fit integration
        fct = lambda x : -expo(x, a, b)
        intr, err = integ.quad(fct, 0, tlim)
        if a < 0 or b < 0:
            intr = None
            inttot = None
        else:
            inttot = intl + intr
    except:
        print("didn't converge")
        return(None)
    if limits == True:
        return(intl, intr, err, bl, a, b, tm, dtl, dtr, tfit, tlim)
    return(inttot)

def integral_plot(n):
    t, I2 = load(n)
    return(integral(t, I2))

def integral_tot(ni=0, nf=1000000):
    t0, Is = loadfile()
    L = []
    c = 0
    for i in range(ni, nf):
        if i/1000 == i//1000:
            print(i)
        I20 = Is[i]
        t, I2 = t0, I20
        I = integral(t, I2)#, newmin = True)
        if I == None:
            c += 1
        else:
            L.append(I)
    return(L, c)



def new_int_dled(t, A, n, ampllim = 9): # search all the peaks of the signal
    # (t, A): signal
    # n: nb of bin-size for the dled delay (dled = "discret derivative")
    # ampllim: min amplitude to select a peak
    L = [0] # list of peaks
    dled = A[:-n] - A[n:] # derivative
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
    L.append(len(A)-1)
    return(L) # return the list of peaks in this form: [0, beginning peak 1, end peak 1, beginning peak 2, end peak 2, ..., 449]

def plot_dled(n, dt):
    t, I = load(n)
    t *= 1e6
    T = new_int_dled(t, I, dt, 9)
    print(T)
    #plt.plot(t, I - 10000, label = 'response signal')
    plt.plot(t[:-dt], I[:-dt] - I[dt:], label = 'dled signal')
    plt.xlabel(r'time ($\mu s$)')
    plt.ylabel('amplitude')
    plt.legend()



def expo2(c):
    return(lambda x, a, b : c - a*np.exp(-b*x))

# for SiPM
def new_int(t, A, graph = False, info = False): # !! time in ten ns (= 1 bin-size) !!
    blmoy = 10000
    A = A.astype('float64') - blmoy # easily treated (especially for the fit)
    if graph == True:
        plt.plot(t*1e6, A, label = 'response signal')
        tplot_tot = []
        A_tot = []
    I = []
    tdled = 2 # dt for the discret derivative
    ampllim = 9 # min amplitude to select a peak
    T = np.array(new_int_dled(t, A, tdled, ampllim))
    if info == True:
        print(T)
    tt = np.array([i for i in range(len(A))]) # time in bin-size
    #param :
    Tmin = 138 # average minimum time
    sigma = 3 # average deviation of the min
    tc = 10 # min time to considere 2 peaks independantly
    tlr = 10 # right limit of the window to search Amin
    dtl, dtr = -2, 1 # left and right limits of direct integration
    tfit = 20 # time window of the fit
    tlim = 100 # time window of the fit integration
    dt = T[1]
    bl = A[T[1]]
    cond = 0 # condition to start the integrations
    for i in range(len(T)//2):
        dt = T[2*i + 1] - T[2*i]
        if info == True:
            print('i = ', i)
            print('dt = ', dt)
            print('cond = ', cond)
        if dt >= tc:
            if cond != 0: # condition to wait for tl to be define
                Am = A[T[2*i-1]:T[2*i]+tlr]
                ttm = tt[T[2*i-1] : T[2*i] + tlr] # time window to find the min of the signal
                Amin = np.min(Am) # (local) min of the signal
                tmin = ttm[Am == Amin][0] # time of the min
                tl = tl[tl <= tmin + dtr] # right limit of the time window for (left) direct integration
                Il = integ.simps(bl - A[tl], tl/100) # left direct int.
                tfit2 = min(tmin + tfit, T[2*i+1])
                tr = tt[tmin+dtr:tfit2] # time window for the fit
                amp = bl - Amin
                if len(tr) >= 5 and amp > 0:
                    tr2 = tr - tr[0]
                    if info == True:
                        print('tmin = ', tmin)
                        print('amp max = ', amp)
                    try:
                        fct_fit = expo2(bl) # fct used for the fit
                        popt, pcov = curve_fit(fct_fit, tr2/100, A[tr], p0 = np.array([amp, 6.8]), bounds =  ([amp/1.5, 6.7], [amp*1.5, 6.8]))
                        a, b = popt
                        tnew = tt[tr[0]:]
                        if graph == True:
                            if cond == 1:
                                fct_fit_tot = fct_fit((tnew - tr[0])/100, a, b)
                            else:
                                fct_fit_tot = fct_fit_tot[tr[0]-tlimplot:] - bl_old + fct_fit((tnew - tr[0])/100, a, b)
                                tplot = tt[tlimplot:tr[0]]
                                tplot_tot += list(tplot/100)
                                A_tot += list(A[tplot])
                            tnew2 = min(tlim, tt[-1]-tr[0])
                            plt.plot(tnew[:tnew2]/100, fct_fit_tot[:tnew2], label = '%sth peak'%i)    
                            tlimplot = tr[0]
                            bl_old = bl
                        fct_int = lambda x : bl - fct_fit(x, a, b) # fct used to calculate the int.
                        Ir, err = integ.quad(fct_int, 0, (tlim-dtr)/100) # right int.
                        if abs(tmin - Tmin) < sigma:
                            I.append(Il + Ir) # total int.
                        A[tnew] += (fct_int((tnew-tr[0])/100)) # substract the fit to the global signal (only from the left time limit of the fit to avoid huge values of the exp)
                        if info == True:
                            print('bl, a, b : ', bl, a, b)
                            print('intl, intr : ', Il, Ir)
                    except:
                        cond = 1
                        if info == True:
                            print('fit failed')
            if i == len(T)//2 + 1:
                break
            if cond == 0:
                bl = np.mean(A[T[2*i+1]-9:T[2*i+1]+1])
            else:
                tlimbl = 20
                tlimbl2 = max(T[2*i+1] - tlimbl , tmin + dtr)
                bl = np.mean(A[tlimbl2:T[2*i+1]+1])
            tl = tt[tt >= T[2*i+1]+dtl] # left limit of the time window for direct integration
            cond += 1
    if graph == True:
        plt.plot(tplot_tot, A_tot)
        plt.xlabel(r'time ($\mu s$)')
        plt.ylabel('amplitude')
        plt.legend()
    return (I)


# for Abalone : almost the same fct with different parameters
def new_int2(t, A, graph = False, info = False): # !! time in ten ns (= 1 bin-size) !!
    blmoy = 12250
    A = A.astype('float64') - blmoy
    if graph == True:
        plt.plot(t*1e6, A, label = 'response signal')
        tplot_tot = []
        A_tot = []
    I = []
#    errela = []
#    errelb = []
    tdled = 5
    ampdled = 20
    T = np.array(new_int_peaks(t, A, tdled, ampdled))
    if info == True:
        print(T)
    tt = np.array([i for i in range(len(A))])
    #param :
    Tmin = 260 # average minimum time
    sigma = 20 # average deviation of the min
    tc = 10 # min time to considere 2 peaks independantly
    tca = 20
    tlr = 10 # right limit of the window to search Amin
    dtl, dtr = -2, 1 # left and right limits of direct integration
    dtra = 15 # for abalone response
    tfit = 20 # time window of the fit
    tfita = 30 # for abalone response
    tlim = 100 # time window of the fit integration
    ampla = 100
    dt = T[1]
    bl = A[T[1]]
    cond = 0 # condition to start the integrations
    bl_old = 0 # if there is a pb with its 1st def
    for i in range(len(T)//2):
        dt = T[2*i + 1] - T[2*i]
        if info == True:
            print('i = ', i)
            print('dt = ', dt)
            print('cond = ', cond)
        if cond != 0:
            Am = A[T[2*i-1]:T[2*i]+tlr]
            ttm = tt[T[2*i-1] : T[2*i] + tlr] # time window to find the min of the signal
            Amin = np.min(Am) # (local) min of the signal
            amp = bl - Amin
            tmin = ttm[Am == Amin][0] # time of the min
        else:
            amp = 0
        if (dt >= tc and amp <= ampla) or (dt > tca and amp > ampla):
            if cond != 0:
                if amp <= ampla:
                    tl = tl[tl <= tmin + dtr] # right limit of the time window for (left) direct integration
                    Il = integ.simps(bl - A[tl], tl/100) # left direct int.
                    tfit2 = min(tmin + tfit, T[2*i+1])
                    tr = tt[tmin+dtr:tfit2] # time window for the fit
                else:
                    tl = tl[tl <= tmin + dtra] # right limit of the time window for (left) direct integration
                    #print('tl : ', tl)
                    Il = integ.simps(bl - A[tl], tl/100) # left direct int.
                    tfit2 = min(tmin + tfita, T[2*i+1])
                    tr = tt[tmin+dtra:tfit2] # time window for the fit
                if info == True:
                    print('tmin = ', tmin)
                    print('amp max = ', amp)
                    print('tr = ', tr)
                if len(tr) >= 5 and amp > 0:
                    tr2 = tr - tr[0]
                    try:
                        fct_fit = expo2(bl) # fct used for the fit
                        if amp > ampla:
                            popt, pcov = curve_fit(fct_fit, tr2/100, A[tr], p0 = np.array([-A[tr[0]], 8]), bounds =  ([-A[tr[0]]/1.5, 7.5], [-A[tr[0]]*1.5, 8.5]))
                        else:
                            popt, pcov = curve_fit(fct_fit, tr2/100, A[tr], p0 = np.array([amp, 6.8]), bounds =  ([amp/1.5, 6.5], [amp*1.5, 7.5]))
                        a, b = popt
                        perr = np.sqrt(np.diag(pcov))
                        errela = (perr[0]/a)
                        errelb = (perr[1]/b)
                        tnew = tt[tr[0]:]
                        if graph == True:
                            if cond == 1:
                                fct_fit_tot = fct_fit((tnew - tr[0])/100, a, b)
                            else:
                                fct_fit_tot = fct_fit_tot[tr[0]-tlimplot:] - bl_old + fct_fit((tnew - tr[0])/100, a, b)
                                tplot = tt[tlimplot:tr[0]]
                                tplot_tot += list(tplot/100)
                                A_tot += list(A[tplot])
                            tnew2 = min(tlim, tt[-1]-tr[0])
                            plt.plot(tnew[:tnew2]/100, fct_fit_tot[:tnew2], label = '%sth peak'%i)    
                            plt.vlines(tl[0]/100, A[tl[0]] - 10, bl + 10)
                            tlimplot = tr[0]
                            bl_old = bl
                        fct_int = lambda x : bl - fct_fit(x, a, b) # fct used to calculate the int.
                        Ir, err = integ.quad(fct_int, 0, (tlim-dtr)/100) # right int.
                        if abs(tmin - Tmin) < sigma and errela < 0.1 and errelb < 0.1: #and amp > 100:
                            I.append(Il + Ir) # total int.
                        A[tnew] += (fct_int((tnew-tr[0])/100)) # substract the fit to the global signal (only from the left time limit of the fit to avoid huge values of the exp)
                        if info == True:
                            print('bl, a, b : ', bl, a, b)
                            print('intl, intr : ', Il, Ir)
                    except:
                        cond = 1
                        if info == True:
                            print('fit failed')
            if i == len(T)//2 + 1:
                break
            if cond == 0:
                bl = np.mean(A[T[2*i+1]-9:T[2*i+1]+1])
            else:
                tlimbl = 20
                tlimbl2 = max(T[2*i+1] - tlimbl , tmin + dtr)
                bl = np.mean(A[tlimbl2:T[2*i+1]+1])
            tl = tt[tt >= T[2*i+1]+dtl] # left limit of the time window for direct integration
            cond += 1
    if graph == True:
        plt.plot(tplot_tot, A_tot)
        plt.xlabel(r'time ($\mu s$)')
        plt.ylabel('amplitude')
        plt.legend()
    return(I)#, errela, errelb)

def new_int_plot(n, graph = True, info = True):
    t, A = load(n)
    I = new_int2(t, A, graph = graph, info = info)
    return(I)

def new_int_tot(ni = 0, nf = 1000000):
    t, As = loadfile(current_file)
    I2 = []
#    Lerrela = []
#    Lerrelb = []
    for i in range(ni, nf):
        if i/100 == i//100:
            print(i)
        I = new_int2(t, As[i])
        I2 += I
#        Lerrela += errela
#        Lerrelb += errelb
    return(I2)#, Lerrela, Lerrelb)#, B1, B2)


def new_int_test1(t, A): # test of the presence of a response peak and the stability of the baseline
    Tmin = 137 # average position of the min
    sigma1 = 10 # average deviation of the min
    tc = 10 # min time to seperate 2  peaks
    tlimbl = 20 # max length on which define a baseline
    ampblmax = 5 # max fluctuation of the baseline
    bl = []
    # test of the presence of a response peak
    T1 = np.array(new_int_dled(t, A, 2, 9))
    T1m = T1[T1 > Tmin - sigma1]
    T1m = T1m[T1m < Tmin + sigma1]
    if len(T1m) <= 0:
        return(False)
    # test of the stability of the baseline
    for i in range(len(T1)//2): # 
        dt = T1[2*i + 1] - T1[2*i]
        if dt > tc:
            tlimbl2 = max(T1[2*i+1] - tlimbl, T1[2*i] + tc)
            bl.append(np.mean(A[tlimbl2:T1[2*i+1]]))
    if abs(bl[-1] - bl[0]) > ampblmax:
#    if np.max(bl)-np.min(bl) > ampblmax:
        return(False)
    return(True)

def new_int_test2(t, A): # test1 + test of the cleanness of the signal (no peaks to close to the main one)
    if new_int_test1(t, A) == False:
        return(False)
    Tmin = 138
    sigma1 = 3
    sigma2l = 20 # left limit of the time window where we don't want any peak
    sigma2r = 25 # right limit of the time window where we don't want any peak
    tt = np.arange(0, len(A), 1)
    tt1, A1 = tt[tt > Tmin - sigma1], A[tt > Tmin - sigma1]
    tt1, A1 = tt1[tt1 < Tmin + sigma1], A1[tt1 < Tmin + sigma1]
    tmin = tt1[A1 == np.min(A1)][0]
    tt2, A2 = tt[tt > tmin - sigma2l], A[tt > tmin - sigma2l]
    tt2, A2 = tt2[tt2 < tmin + sigma2r], A2[tt2 < tmin + sigma2r]
    T2 = np.array(new_int_dled(0, A2, 2, 5))
    if len(T2) > 4: # = more than 1 peak (the main one)
        return(False)
    return(True)

def new_int_testA(t, A): # test for the Abalone: test only the baseline fluctuation
    tdled = 2
    ampdled = 9
    tc = 10
    tlimbl = 20
    ampblmax = 5
    bl = []
    T1 = np.array(new_int_dled(t, A, tdled, ampdled))
    for i in range(len(T1)//2): # 
        dt = T1[2*i + 1] - T1[2*i]
        if dt > tc:
            tlimbl2 = max(T1[2*i+1] - tlimbl, T1[2*i] + tc)
            bl.append(np.mean(A[tlimbl2:T1[2*i+1]]))
    if abs(bl[-1] - bl[0]) > ampblmax:
    #if np.max(bl)-np.min(bl) > ampblmax:
        return(False)
    return(True)

def new_int_test_plot(n):
    t, A = load(n)
    A -= 10000
    return(new_int_test2(t, A)) #choose if you want only test1 or test1+test2 or testA

def new_int_test_tot(ni = 0, nf = 1000000):
    test_tot = []
    t, As = loadfile(current_file)
    for i in range(ni, nf):
        if i/1000 == i//1000:
            print(i)
        test = new_int_testA(t, As[i]) # choose test2 or testA
        if test == True:
            test_tot.append(i)
    return(test_tot) # return the list of all the numers of events whose the tests respond True


def new_int_list(L):
    # L: list of numers of events which will be analyze
    t, As = loadfile(current_file)
    I2 = []
#    Lerrela = []
#    Lerrelb = []
    c=0
    for i in list(L):
        if c/10 == c//10:
            print(c)
        c+=1
        I = new_int2(t, As[i])
        I2 += I
#        Lerrela += errela
#        Lerrelb += errelb
    return(I2)#, Lerrela, Lerrelb)



def savefile(name_file, file):
    f = open('results/Abalone/Abalone_22kV/' + name_file, 'w') #adjust the path to what you want
    for i in file:
        f.write('{}'.format(i))
        f.write('\n')
    f.close()



def histo_plot(I, n = 0.1, xmin = 0, xmax = 300, ymin = 0, ymax = 500): # make a simple histogram
    plt.hist(I, bins = np.arange(xmin, xmax, n))
    plt.xlabel('area '+r'($\times 10^{-6}$)')
    plt.ylabel('number of events')
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.legend()
    plt.show()

def histo(a, I, n=1e-9, xmax = None, k=None, graph=True, log = False): # make a (multi-)histogram or return the values of the x and y-axis of the histogram
    ## for simple histogram, use histo_plot
    # a = 'd' (for delay) or 'i' (for integration) or the name of the abscisse if neither delay nor area
    # I: list of values of delay or area
    # n: bin-size of the histogram
    # xmax: maximum value of the (x-axis of the) histogram, if None: xmax = max(I) +n
    # k = height or number of smoothing
    # graph: for plotting the histogram
    # log for having the y-axis on log scale
    I = np.array(I)
    m = np.min(I)
    if xmax == None:
        xmax = np.max(I) + n
    X = np.arange(m, xmax, n) # x-axis of the histogram (delay or area values)
    y = np.zeros(len(X)) # y-axis of the histogram (number of events)
    deleted = 0
    for i in I:
        if i/1000 == i//1000:
            print(i)
        if i > xmax:
            deleted += 1
        else:
            for j in range(len(y)):
                if i <= m + j*n :
                    y[j] += 1
                    break
    print('nb of deleted data : ', deleted)
    Y = np.array(y)
    if graph == True:
        plt.plot(X, Y, marker = '.')
        if a == 'i':
            plt.xlabel(r'area ($\times 10^{-9}$)')
        else:
            plt.xlabel('{}'.format(a))
        plt.ylabel('number of events')
        #plt.ylim(Y[-1], np.max(Y))
        if log == True:
            plt.yscale('log')
        plt.legend()
        plt.show()
    else:
        return(X, Y)



def gaussian(x, a, b, c):
    return(a*np.exp(-b*(x-c)**2))

def max_histo(x, y, graph = False): # a and b are the limits of the fit (in microsec)
    A = []
    erA = []
    B = []
    erB = []
    Xmax = [] # list of the place of the max of the fit of each peak
    erXmax = []
    Xmax0 = 2500
    xx = np.arange(0, Xmax0, 1) # integer list of abscisse
    Xmoy = 211 # average place of the maximum of the 1st peak #178 for l4b, 211 for l5a
    width = 50 # average width of each peak
    x = np.array(x)
    y = np.array(y)
    for i in range(1, Xmax0//Xmoy):
        print(i)
        xmoy = i*Xmoy # average place of the min of the i-th peak
        ymoy = y[xmoy]
        t1 = xx[xmoy -width:xmoy + width] # time window used for the fit
        x1 = x[t1]
        popt, pcov = curve_fit(gaussian, x1, y[t1], p0 = [ymoy, 3.5, xmoy/100], bounds = ([ymoy/2, 1, (xmoy-width)/100], [ymoy*2, 10, (xmoy+width)/100]))
        a, b, xmax = popt
        era, erb, erc = np.sqrt(np.diag(pcov))
        print(popt)
        if graph == True:
            if i == 0:
                plt.plot(x1, gaussian(x1, *popt), color = 'r', label = 'gaussian fit')
            else:
                plt.plot(x1, gaussian(x1, *popt), color = 'r')
            #plt.vlines(xmax, 0, a*1.1, label = '{} -> {}'.format(round(xmax, 2), round(xmax/i, 2)))
        else:
            A.append(a)
            erA.append(era)
            B.append(b)
            erB.append(erb)
            Xmax.append(xmax)
            erXmax.append(erc)
    if graph == True:
        plt.xlabel(r'area ($\times 10^{-6}$)')
        plt.ylabel('number of events')
        plt.xlim(0, 25)
        plt.legend()
        print(A, erA)
        print(B, erB)
        print(Xmax, erXmax)
    else:
        return(A, erA, B, erB, Xmax, erXmax)



from scipy import stats


def poisson_binomial(x, l, P, c): # if you don't understand where does this function come from, ask to Yelle ;)
    if type(x) is int:
        return(c*sum([stats.poisson.pmf(x-i, l)*stats.binom.pmf(i, x-i, P) for i in range(x//2)]))
    else:
        result = np.zeros(np.size(x))    
        for npe in range(np.size(x)):
            for ndpe in range(int(npe//2)+1):
                result[npe] += stats.poisson.pmf(x[npe]-ndpe, l)*stats.binom.pmf(ndpe, x[npe]-ndpe, P)
    return(c*result)

def pe_peak_amplitude(A, erA): # fit of the amplitude of the peaks
    pe = np.arange(1, len(A)+1, 1)
    plt.errorbar(pe, A, yerr = erA, marker = '.')
    fct_fit = poisson_binomial
    popt, pcov = curve_fit(fct_fit, pe, A, p0 = [3, 0.1, 2500], sigma = np.array(erA)/A)
    param, param2, const = popt
    erl, erp, erc = np.sqrt(np.diag(pcov))
    print(param, erl, param2, erp, const, erc)
    plt.plot(pe, fct_fit(pe, param, param2, const), label = r'fit : $A_{0} \times Poisson(\lambda) \ast Binomial(p)$' + '\n' + r'$A_{0} = %s $ ; '%int(const) + r'$ \lambda = %s $ ; '%round(param, 2) + r'$p = %s $'%round(param2, 2))
    plt.xlabel('pe peak')
    plt.ylabel('amplitude')
    plt.legend()

def pe_peak_width(W, erW, A): # fit of the width of the peaks
    pe = np.arange(1, len(W)+1, 1)
    W = np.array(W)
    A = np.array(A)
    S = 1/np.sqrt(2*W)
    erS = erW/(2*W)**(3/2)
    plt.errorbar(pe, S, yerr = erS, marker = '.')
    fct_fit = lambda x, a, b : (a**2 + x*b**2)**(1/2)
    popt, pcov = curve_fit(fct_fit, pe, S, p0 = [10, 1], sigma = erS/S)
    sigma0, sigmape = popt
    ersig0, ersigpe = np.sqrt(np.diag(pcov))
    print(sigma0, ersig0, sigmape, ersigpe)
    plt.plot(pe, fct_fit(pe, sigma0, sigmape), label = r'fit : $(\sigma_{noise}^{2} + n_{pe} \times \sigma_{pe}^{2})^{1/2}$' + '\n' + r'$\sigma_{noise} = %s $ ; '%round(sigma0, 2) + r'$\sigma_{pe} = %s $'%round(sigmape, 2))
    plt.xlabel('pe peak')
    plt.ylabel('width')
    plt.legend()
 
def pe_peak_position(P, erP, A): # fit of the position of the peaks (what is really interesting !)
    pe = np.arange(1, len(P)+1, 1)
    A = np.array(A)
    plt.errorbar(pe, P, yerr=erP, marker = '.')
    fct_fit = lambda x, a : a*x
    popt, pcov = curve_fit(fct_fit, pe, P, p0 = [2], sigma = erP/A)
    param = popt[0]
    era = np.sqrt(np.diag(pcov))[0]
    print(param, era)
    plt.plot(pe, fct_fit(pe, param), label = r'fit : $\mu_{0} \times n_{pe}$' + '\n' + r'$\mu_{0} = %s $'%round(param, 2))
    plt.xlabel('pe peak')
    plt.ylabel('amplitude')
    plt.legend()




current_file = '/media/suxess/DATADISK2/daq_data/DetectorPhysics/test190912/raw/26V_1.54V/26V_1.54V_Module_0_0.root'

t, L2 = loadfile()
print(t,L2)


















