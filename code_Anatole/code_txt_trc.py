#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 10:11:47 2019

@author: anatole
"""


from trcreader import Trc
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integ
from scipy.optimize import curve_fit



def nbr(n): #change nbs into strings for the file names
    if n < 10:
        return('000%s'%n)
    elif n < 100:
        return('00%s'%n)
    elif n < 1000:
        return('0%s'%n)
    else:
        return('%s'%n)


#files names
# for each file: n (between 1 and 6552) is the numer of the event and c (= 1 or 2) is for the emission signal (1) or the response (2)

#txt files

def fichier1(n, c): #laser 1 or 2: "Laser_zoomed_in_ABALONEOnly_100psI" for 1 and "Laser_zoomed_in_ABALONEOnly_100psII" for 2 (change I into II)
    return('/media/anatole/MAXTOR/ABALONE/ABALONE_measurements_fast_scope/Laser_zoomed_in_ABALONEOnly_100psI/C{}Laser_Zoomed_in_ABALONETriggerOnly100ps0{}.txt'.format(c, nbr(n)))

def fichier2(n, c): #LED 1 or 2 (I or II)
    return('/media/anatole/MAXTOR/ABALONE/ABALONE_measurements_fast_scope/LED_1.62V_LEDTriggerOnly_ASCII_I/C{}LED_1.62V_LEDTriggerOnly100ps0{}.txt'.format(c, nbr(n)))

#trc files

def fichier3(n, c): #LED+ 1 or 2 (I or II)
    return('/media/anatole/MAXTOR/ABALONE/ABALONE_measurements_fast_scope/LED_1.65V_LEDTriggerOnlyI/C{}LED_1.65V_LEDTriggerOnly100ps0{}.trc'.format(c, nbr(n)))

def fichier4(n, c): #LED 3 only ! (4 empty)
    return('/media/anatole/MAXTOR/ABALONE/ABALONE_measurements_fast_scope/LED_1.62V_LEDTriggerOnly25psI/C{}LED_1.62V_LEDTriggerOnly25ps0{}.trc'.format(c, nbr(n)))

def fichier5(n, c): #LED 5 or 6 (I or II)
    return('/media/anatole/MAXTOR/ABALONE/ABALONE_measurements_fast_scope/LED_1.62V_ABALONETriggerOnly100psI/C{}LED_1.62V_ABALONETriggerOnly100ps0{}.trc'.format(c, nbr(n)))

def fichier6(n, c): #LED 7 or 8 ( None or II)
    return('/media/anatole/MAXTOR/ABALONE/ABALONE_measurements_fast_scope/LED_1.62V_LEDTriggerOnly/C{}LED_1.62V_LEDTriggerOnly100ps0{}.trc'.format(c, nbr(n)))

def fichier7(n, c): #laser 3, 4, 5 or 6 (I, II, III or IV)
    return('/media/anatole/MAXTOR/ABALONE/ABALONE_measurements_fast_scope/Long_run_ABALONEOnlyTrigger100psIV/C{}ABALONEOnlyTrigger100ps1MHz0{}.trc'.format(c, nbr(n)))


def current_file(n, c): #choose the file you want work with !!don't forget to change the laod function if you change the type of file (txt or trc)!!
    return(fichier1(n, c))


#load files

#txt files

def loadtext(n):
    M = np.loadtxt(current_file(n, 1), delimiter=',', skiprows=5) # array with time and emission signal
    N = np.loadtxt(current_file(n, 2), delimiter=',', skiprows=5) # array with time and response signal
    t = M[:, 0] # time
    I1 = M[:, 1] # emission signal
    I2 = N[:, 1] # response signal
    return(t, I1, I2)

def loadtext2(n): # load only response signal (and time)
    M = np.loadtxt(current_file(n, 2), delimiter=',', skiprows=5)
    return(M[:, 0], M[:,1])

#trc files

trc = Trc()

def loadtrc(n):
    t, I1, d1 = trc.open(current_file(n, 1))
    t2, I2, d2 = trc.open(current_file(n, 2))
    #t0, I10, I20 = t[t <= 1.5e-6], I1[t <= 1.5e-6], I2[t <= 1.5e-6]
    t0, I10, I20 = t, I1, I2
    return(t0, I10, I20)

def loadtrc2(n):
    t, I2, d = trc.open(current_file(n, 2))
    #t0, I20 = t[t <= 1.5e-6], I2[t <= 1.5e-6]
    return(t, I2)


def load(n): #select the line you need (if you work on a txt or trc file)
    return(loadtext(n))
    #return(loadtrc(n))

def load2(n): #same
    return(loadtext2(n))
    #return(loadtrc2(n)) 



def plotI(n): #plot the emission and response signal
    t, I1, I2 = load(n)
    plt.plot(t*1e9, I1, label = 'laser signal')
    plt.plot(t*1e9, I2*20, label = r'response signal ($\times 20$)')
    plt.xlabel(r'time ($ns$)')
    plt.ylabel('amplitude')
    plt.legend()

def plotI2(n): #plot only the response signal
    t, I2 = load2(n)
    plt.plot(t*10**9, I2, label = r'response signal')
    plt.xlabel(r'time ($ns$)')
    plt.ylabel('amplitude')
    plt.legend()



def deriveeI(t, I): #calculate the derivative of a function
    return(t[:-1], (I[1:] - I[:-1])/(t[1:] - t[:-1]))

def deriveeI_plot(n): #plot the derivative of the emission signal of the current file
    # n is the numer of the event
    t, I1, I2 = load(n)
    dt, dI1 = deriveeI(t, I1)
    plt.plot(dt, dI1)



def smoothI(t, I, n): #(basic) smoothing function
    # t, I is the x, y axis of your function
    # n is the number of points you want to average
    sI = [] # smoothed y axis
    st = [] # smoothed x axis
    for i in range(int(len(t)/n)):
        sI.append(np.mean(I[i*n:(i+1)*n]))
        st.append(np.mean(t[i*n:(i+1)*n]))
    return(np.array(st), np.array(sI))

def smoothIplot(n, smooth, graph=False): # plot the smooth function of the response signal of the current file
    t, Ir = load2(n)
    st, sIr = smoothI(t, Ir, smooth)
    plt.plot(st, sIr)



def test1(t, I1): #test on the emission signal (return True if there is a trigger peak)
    dt, dIl = deriveeI(t, I1)
    dIlm = np.max(dIl)
    if dIlm < 1.5e9: # test if there is a triggering peak
        print('e1')
        return(False)
    tm = dt[dIl == dIlm][0]
    t1 = dt[dt < tm]
    dIl1 = dIl[dt < tm]
    t2 = t1[dIl1 < 1e9]
    if np.size(t2) == 0: # test if this peak isn't too close to the left limit of the acquisition time window
        print('e1b')
        return(False)
    return(True)

def test2(t, I2): #test on the response signal (return True if there is a response peak)
    st, sI2 = smoothI(t, I2, 1)#5)
    sst, ssI2 = smoothI(t, I2, 6)#00)
    if ssI2[0] - ssI2[-1] > 0.01:
         tlim = sst[ssI2 > ssI2[0] - 0.01][-1]
         t, I2 = t[t < tlim], I2[t < tlim]
    sstm = sst[ssI2 == np.min(ssI2)][0]
    if np.min(ssI2) > - 0.02: # presence of the response peak
        print('e2')
        return(False)
    if np.size(sI2[st <= sstm - 1.5e-7]) == 0: # possibility to define a baseline
        print('e2b')
        return(False)
    return(True)



def delay(t, I1, I2, height=0.05, smooth=5): #compute the delay between the trigger (I1) and the response (I2)
    # t, I1, I2: trigger/response signal
    # height: height of the threshold to determine T2 (0 = baseline, 1 = minimum)
    dt, dI1 = deriveeI(t, I1)
    dt1 = dt[dt < dt[dI1 == np.max(dI1)][0]]
    T1 = dt1[dt1 < 10**9][-1] # T1 is defined as the first time the derivative of the trigger signal reach 1e9
    #
    st, sI2 = smoothI(t, I2, smooth) # slightly smoothed emission signal, used instead of the real signal to get rid of the fluctuation without loosing too much time precision
    sst, ssI2 = smoothI(t, I2, 600) # strongly smoothed emission signal, used to find the response peak without dealing with large noise
    ssI2m = np.min(ssI2)
    tlim1 = sst[ssI2 == ssI2m][0] - 1e-7
    sImoy = sI2[st <= tlim1 - 0.5e-7]
    moy = np.mean(sImoy) # baseline
    sI21 = sI2[st > tlim1]
    st1 = st[st > tlim1]
    tlim2 = sst[ssI2 == ssI2m][0] + 1e-7
    sI21 = sI21[st1 < tlim2]
    sI2m = np.min(sI21) # min of the (slightly smoothed) emission signal
    sI3 = sI2[st < st[sI2 == sI2m][0]]
    st3 = st[st < st[sI2 == sI2m][0]]
    if np.size(st3[sI3 - moy > height*(sI2m - moy)]) == 0: #check if we can define T2
        return(None)
    T2 = st3[sI3 - moy > height*(sI2m - moy)][-1] # T2 is defined as the last time (before the minimum) the response signal is above the threshold
    return(T2 - T1) # return the delay

def delay_current(n, a=0.1, s=5): # return the delay between the trigger and response signal of the current file
    t, I1, I2 = load(n)
    return(delay(t, I1, I2, height=a, smooth=s))

def delay_tot(ni = 0, nf = 6553): # compute the delay of an entire directory
    # ni and nf are the first and last(+1) numer of event treated
    L = [] # list of delays
    c=0 # number of events without any delay computed
    for i in range(ni, nf):
        if i/10 == i//10:
            print(i) # just to see were we are...
        t, I1, I2 = load(i)
        if test1(t, I1) == False or test2(t, I2) == False:
            c += 1
        else:
            dt = delay(t, I1, I2)
            if dt == None:
                c += 1
            else:
                L.append(dt)
    return(L, c) # return the list of delays and the number of events with no delay found

def delay_tot2(l): # delays for different smoothing or height
    # l: list of height or number of smooth
    c = 0 # number of events without trigger or response signal (so we can't define any delay)
    c2 = [0 for i in l] # number of events (for each value l) for which we can't define any delay
    L = [[] for i in l] # list of list of delays for each value of l
    for i in range(6553):
        if i/10 == i//10:
            print(i)
        t, I1, I2 = load(i)
        if test1(t, I1) == False or test2(t, I2) == False:
            c += 1
        else:
            for j in range(len(l)):
                if l[0] < 1:
                    dt = delay(t, I1, I2, l[j], 5)
                else:
                    dt = delay(t, I1, I2, 1/20, l[j])
                if dt == None:
                    c2[j] += 1
                L[j].append(dt)
    c2 = np.array(c2)+c
    return(L, c2)



def expo(x, a, b):
    return(-a*np.exp(-b*x))

def integral(t, I, info = False): # return the area of the response peak of the signal (t, I)
    # if info == True: return all the specific value used in this function
    t = np.array(t)*1e6 # time
    I = np.array(I) # signal
    bl = 10000 # base line (12350 for L1, L2, l, and 10000 for l2, l3)
    I = I - bl # signal shifted such that bl = 0
    ##find min
    Tm = 1.37 # average minimum position
    dt = 0.1 # average deviation of the min
    t1, I1 = t[t > Tm - dt], I[t > Tm - dt]
    t2, I2 = t1[t1 < Tm + dt], I1[t1 < Tm + dt]
    tm = t2[I2 == np.min(I2)][0]
    t -= tm # time shifted such that the min is at t=0
    ##cst direct integration
    dtl = -0.03 #cst left time limit for direct integration
    dtr = 0.02 #cst right time limit of direct integration
    tl, Il = t[t <= dtr], I[t <= dtr]
    intl = -1 * integ.simps(Il[tl >= dtl], tl[tl >= dtl]) # direct integration
    ##fit and integration
    tr, Ir = t[t >= dtr], I[t >= dtr]
    tfit = 0.20 # cst time window for fit
    try:
        popt, pcov = curve_fit(expo, tr[tr < tfit], Ir[tr < tfit], p0 = np.array([200, 1]))
        a, b = popt
        tlim = 1 # time limit for fit integration
        fct_int = lambda x : -expo(x, a, b)
        intr, err = integ.quad(fct_int, 0, tlim)
        if a < 0 or b < 0:
            intr = None
            inttot = None
        else:
            inttot = intl + intr
    except:
        print("fit failed")
        return(None)
    if info == True:
        return(intl, intr, err, bl, a, b, tm, dtl, dtr, tfit, tlim)
    return(inttot)

def int_plot(t, I2, info = False): # plot the the response peak and all the values needed for the integration
    ymin = np.min(I2)
    try:
        Il, Ir, err, bl, a, b, tm, dtl, dtr, tfit, tlim = integral(t, I2, limits = True)
        if info == True:
            print(Il, Ir, err, a, b, tm, dtl, dtr, tfit, tlim)    
    except:
        print("fit failed")
        bl = 10000 #12350
        tm = 1.3
        dtl = -0.03
        dtr = 0.02
        tfit = 0.15
        tlim = 1
    t = t*1e6 - tm
    plt.plot(t, I2, label = r'response signal')
    plt.xlabel(r'time ($\mu s$)')
    plt.ylabel('amplitude')
    t0 = np.linspace(0, t[-1], 100)
    plt.plot(t0, bl + expo(t0, a, b), label = 'fit f(x) = baseline - a*exp(-b*x): \n a = {} ; b = {}'.format(round(a, 1), round(b, 1)))
    plt.hlines(bl, t[0], t[-1], colors = 'r', label = 'baseline')
    plt.vlines(0, ymin - 10, bl + 10, label = r'$t_{min}$')
    plt.vlines(dtl, I2[t >= dtl - 1e-3][0] - 10, bl + 10, colors = 'g', label = 'integration limits')
    plt.vlines(dtr, I2[t >= dtr - 1e-3][0] - 10, bl + 10, colors = 'g')
    plt.vlines(tfit, I2[t >= tfit - 1e-3][0] - 10, bl + 10, colors = 'c', label = 'fit limit')
    plt.vlines(tlim, I2[t >= tlim - 1e-3][0] - 10, bl + 10, colors = 'g')
    plt.legend()

def integral_current(n): #return the area of the response peak of the nth event of the current file
    t, I2 = load2(n)
    return(integral(t, I2))

def integral_tot(ni=0, nf=6553): # compute the area of the response peak for all events of a directory
    L = [] # list of the areas of each event
    c = 0 # number of event with no area found
    for i in range(ni, nf):
        if i//10 == i/10:
            print(i)
        t, I2 = load(i)
        I = integral(t, I2)
        if I == None:
            c += 1
        else:
            L.append(I)
    return(L, c)



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
    if graph == True:
        if k == None:
            plt.plot(X, y, marker = '.')
        elif k >= 1:
            plt.plot(X*1e9, y, marker = '.', label = 'smooth {} pts'.format(k))
        else:
            plt.plot(X*1e9, y, marker = '.', label = 'limit at {}'.format(k))
        if a == 'd':
            plt.xlabel(r'delay ($ns$)')
        elif a == 'i':
            plt.xlabel(r'area ($\times 10^{-9}$)')
        else:
            plt.xlabel('{}'.format(a))
        plt.ylabel('number of events')
        #plt.ylim(y[-1], np.max(y))
        if log == True:
            plt.yscale('log')
        plt.legend()
        plt.show()
    else:
        return(X, y)

def histo_list(L, l, n): # plot the histogram of L for different value of height or smoothing
    # L: list of the values of delay for the histogram
    # l: list of the height or smooth
    # n: bin-size of the histogram
    for i in range(len(L)):
        print(i)
        histo('d', L[i], n = n, k = l[i])


def gaussian(x, a, b, c):
    return(a*np.exp(-b*(x-c)**2))

def histo_fit(x, y, fit = True, a = 1, b = 15, graph = False): #gaussian fit of a histogram
    # a and b are the limits of the fit (in microsec)
    x = np.array(x)*10**6
    y = np.array(y)/1000 # to have lower values for the fit
    x, y = x[x > a], y[x > a]
    x, y = x[x < b], y[x < b]
    popt, pcov = curve_fit(gaussian, x, y)
    xm = popt[2]
    if graph == False:
        return(xm)
    plt.plot(x, y, marker = '.', linestyle = '', label = 'data')
    X = np.linspace(a, b, num  = 100)
    plt.plot(X, gaussian(X, *popt), label = 'gaussian fit \n max at %s'%xm)
    plt.xlabel(r'area ($\times 10^{-6}$)')
    plt.ylabel('number of events (log)')
    plt.legend()







































