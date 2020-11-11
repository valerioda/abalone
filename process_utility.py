import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integ
from scipy.optimize import curve_fit 
import pandas as pd
import scipy.integrate as integ

def read_file( filename, samples = 1024 ):
    data = np.fromfile(filename,  dtype=np.int16)
    n = len(data)/samples
    print(filename,'Number of events',n)
    data = np.array(np.array_split(np.array(data),n))
    return data


def plot_waveform( dts, data ):
    plt.plot(dts,data)
    plt.xlabel('time (ns)')
    plt.ylabel('channels')


def plot_waveforms( dts, data, events = 5 ):
    plt.figure(figsize=(12,6))
    for i in range(events):
        plt.plot(dts,data[i][:])
    plt.xlabel('tims (ns)')
    plt.ylabel('channels')
    

def plot_waveforms2( dts, data, events = 5 ):
    fig, axs = plt.subplots(events, figsize=(12,10))
    for i in range(events):
        axs[i].plot(dts,data[i][:])
        axs[i].set(xlabel='time (ns)', ylabel='channels')

        
def plot_waveforms_comparison( dts, data1, data2, events = 5 ):
    fig, axs = plt.subplots(events,2,  figsize=(15,2*events) )
    for i in range(events):
        axs[i,0].plot(dts,data1[i][:], color='red')
        axs[i,0].set(xlabel='time (ns)', ylabel='channels')
        axs[i,1].plot(dts, data2[i][:], color='blue')
        axs[i,1].set(xlabel='time (ns)', ylabel='channels')


def derive(t, wf): #calculate the derivative of a function
    return(t[:-1], (wf[1:] - wf[:-1])/(t[1:] - t[:-1]))


def smooth(t, wf, n): #(basic) smoothing function
    # t, wf is the x, y axis of your function
    # n is the number of points you want to average
    sw = [] # smoothed y axis
    st = [] # smoothed x axis
    for i in range(int(len(t)/n)):
        sw.append(np.mean(wf[i*n:(i+1)*n]))
        st.append(np.mean(t[i*n:(i+1)*n]))
    return(np.array(st), np.array(sw))


def expo(x, a, b):
    return -a*np.exp(-b*x)


def integral(t, wf, dtl = -1, dtr = 5, tfit = 1000, tlim = 2000, info = False, plot = False):
    #dtl and dtr =  time limit for direct integration
    #tfit = time window for fit
    #tlim = time limit for fit integration
    bl = wf.max() #baseline
    ss = len(wf)
    t = t[int(ss/2-ss/4):int(ss/2+ss/4)]-t[int(ss/4)]
    wf = wf[int(ss/2-ss/4):int(ss/2+ss/4)]
    wf0 = wf - bl #baseline subtraction
    Tm = t[wf.argmin()] #minimum position
    dt = 50 # average deviation of the min
    t1, wf1 = t[t > Tm - dt], wf0[t > Tm - dt]
    t2, wf2 = t1[t1 < Tm + dt], wf1[t1 < Tm + dt]
    tm = t2[wf2 == np.min(wf0)][0]
    t0 = t - tm # time shifted such that the min is at t=0
    tl, wfl = t0[t0 <= dtr], wf0[t0 <= dtr]
    intl = -1 * integ.simps(wfl[tl >= dtl], tl[tl >= dtl]) # direct integration
    ##fit and integration
    tr, wfr = t0[t0 >= dtr], wf0[t0 >= dtr]
     # cst time window for fit
    try:
        popt, pcov = curve_fit(expo, tr[tr < tfit], wfr[tr < tfit],
                               p0 = np.array([200, 1]))
        a, b = popt
        fct_int = lambda x : -expo(x, a, b)
        if plot:
            plt.figure(figsize=(12,6))
            plt.plot(t/1000, wf, label='SiPM signal')
            te = t[:int(tfit/t[1])]
            plt.plot((te+tm)/1000, bl + expo(te, a, b),
                     label=f'fit f(x) = baseline - a*exp(-b*x):\n a = {a:.2f}, b = {b:.2f}')
            plt.hlines(bl, t[0]/1000, t[-1]/1000, colors = 'r', label = 'baseline')
            plt.vlines(tm/1000, np.min(wf) - 10, bl + 10, label = r'$t_{min}$')
            plt.vlines((tm+dtl)/1000, wf[t >= dtl][0]-10, bl+10, colors = 'g',
                       label = 'integration limits')
            plt.vlines((tm+dtr)/1000, wf[t >= dtr][0]-10, bl+10, colors = 'g')
            plt.vlines((tm+tfit)/1000, wf[t >= tfit][0]-10, bl+10, colors = 'c',
                       label = 'fit limit')
            plt.vlines((tm+tlim)/1000, wf[t >= tlim][0]-10, bl+10, colors = 'g')
            plt.xlabel(r'time ($\mu s$)')
            plt.ylabel('amplitude')
            plt.legend()
        intr, err = integ.quad(fct_int, 0, dtr)
        if a < 0 or b < 0:
            intr = None
            inttot = None
        else:
            inttot = intl + intr
    except:
        print("fit failed")
        return None
    if info == True:
        return intl, intr, err, bl, a, b, tm, dtl, dtr
    return inttot

def histo_plot(I, xmin = 0, xmax = 10000, binning = 200): # make a simple histogram
    plt.figure(figsize=(12,6))
    plt.hist(I, bins = np.linspace(xmin, xmax, binning))
    plt.xlabel('area')
    plt.ylabel('number of events')


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

    
def search_peaks(t, wf, n, ampllim = 9, plot = False): # search all the peaks of the signal
    # n: nb of bin-size for the dled delay (dled = "discret derivative")
    # ampllim: min amplitude to select a peak
    L = []
    dled = wf[:-n] - wf[n:] # derivative
    if plot:
        plt.plot(t[:-n]/1000, wf[:-n] - wf[n:], label = 'dled signal')
        plt.xlabel(r'time ($\mu s$)')
        plt.ylabel('amplitude')
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
    #L.append(len(wf)-1)
    return L


def expo2(c):
    return(lambda x, a, b : c - a*np.exp(-b*x))


# for SiPM
def integral_dled( wf, peaks_list, dtl = -2, dtr = 1,
                  tfit = 20, tlim = 100, tc = 5, tlr = 10, plot=False):
    # tlim = time window of the fit integration
    # tfit = time window of the fit
    # tc = min time to consider 2 peaks independently
    # tlr = right limit of the window to search Amin
    inttot = 0
    blmoy = wf.max()
    wf0 = wf - blmoy
    tt = np.array([i for i in range(len(wf))]) # time in bin-size
    tplot_tot = []
    A_tot = []
    cond = 0 # condition to start the integrations
    bl = wf0[peaks_list[0]]
    for i in range(len(peaks_list)//2):
        dt = peaks_list[2*i + 1] - peaks_list[2*i]
        if dt >= tc:
            if cond != 0: #condition to wait for tl to be define
                Am = wf0[peaks_list[2*i]:peaks_list[2*i+1] + tlr]
                ttm = tt[peaks_list[2*i]:peaks_list[2*i+1] + tlr]
                Amin = np.min(Am) #local min of the signal
                tmin = ttm[Am == Amin][0] #time of the min
                #tl = t[t <= tmin + dtr] #right limit of the time window for (left) direct integration
                #Il = -1*integ.simps(bl - wf0[tl], tl/100) # left direct integration
                tl, wfl = tt[tt <= tmin+dtr], wf0[tt <= tmin+dtr]
                Il = -1*integ.simps(wfl[tl >= dtl], tl[tl >= dtl]/100) # left direct integration
                #tfit2 = min(tmin + tfit, peaks_list[2*i+1])
                tr = tt[tmin+dtr:tmin+tfit] # time window for the fit
                amp = bl - Amin
                if len(tr) >= 5 and amp > 0:
                    tr2 = tr - tr[0]
                    #print('tmin:',tmin,'bl:',bl,'amp max:',amp)
                    try:
                        fct_fit = expo2(bl) # fct used for the fit
                        popt, pcov = curve_fit(fct_fit, tr2/100, wf0[tr],
                                                p0 = np.array([amp, 6.8]),
                                                bounds =  ([amp/1.5, 6.7], [amp*1.5, 6.8]))
                        a, b = popt
                        tnew = tt[tr[0]:]
                        if plot:
                            if cond == 1:
                                fct_fit_tot = fct_fit((tnew-tr[0])/100,a,b)
                            else:
                                fct_fit_tot = fct_fit_tot[tr[0]-tlimplot:] - bl_old + fct_fit((tnew - tr[0])/100, a, b)
                                tplot = tt[tlimplot:tr[0]]
                                tplot_tot += list(tplot/100)
                                A_tot += list(wf[tplot])
                            tnew2 = min(tlim, tt[-1]-tr[0])
                            plt.plot(tnew[:tnew2]/100, fct_fit_tot[:tnew2],label = '%sth peak'%i)    
                            tlimplot = tr[0]
                            bl_old = bl
                        fct_int = lambda x : bl - fct_fit(x, a, b)
                        Ir, err = integ.quad(fct_int, 0, (tlim-dtr)/100)
                        if abs(tmin - len(wf)/2) < 11:
                            inttot = Il + Ir
                    #wf0[tnew] += (fct_int((tnew-tr[0])/100)) #substract the fit to the global signal (only from the left time limit of the fit to avoid huge values of the exp)
                    #print('bl, a, b : ', bl, a, b)
                    #print('intl, intr : ', Il, Ir)
                    except:
                        cond = 1
                        #print('fit failed')
            if i == len(peaks_list)//2 + 1: break
            if cond == 0:
                bl = np.mean(wf0[peaks_list[2*i]-9:peaks_list[2*i]+1])
            else:
                tlimbl = 20
                tlimbl2 = max(peaks_list[2*i]-tlimbl, tmin + dtr)
                bl = np.mean(wf0[tlimbl:peaks_list[2*i+1]+1])
            tl = tt[tt >= peaks_list[2*i+1]+dtl] # left limit of the time window for direct integration
            cond += 1
    if plot:
        #plt.plot(tplot_tot, A_tot)
        plt.plot(tt/100, wf0, label = 'SiPM signal')
        plt.xlabel(r'time ($\mu s$)')
        plt.ylabel('amplitude')
        plt.legend()
    if inttot is not 0: return inttot