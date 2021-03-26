import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.integrate as integ
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

def read_file( filename, samples = 1024 ):
    data = np.fromfile(filename,  dtype=np.int16)
    n = len(data)/samples
    print(filename,'Number of events',n)
    data = np.array(np.array_split(np.array(data),n))
    return data


def plot_waveform( data ):
    dts = np.arange(0,data.shape[1]*10,10)
    plt.plot(dts,data)
    plt.xlabel('time (ns)',ha='right',x=1)
    plt.ylabel('channels',ha='right',y=1)


def plot_waveforms( data, events = 5 ):
    dts = np.arange(0,data.shape[1]*10,10)
    plt.figure(figsize=(12,6))
    for i in range(events):
        plt.plot(dts,data[i][:])
    plt.xlabel('tims (ns)',ha='right',x=1)
    plt.ylabel('channels',ha='right',y=1)
    

def plot_waveforms2( data, events = 5 ):
    dts = np.arange(0,data.shape[1]*10,10)
    fig, axs = plt.subplots(events, figsize=(12,10))
    for i in range(events):
        axs[i].plot(dts,data[i][:])
        axs[i].set(xlabel='time (ns)', ylabel='channels')

        
def plot_waveforms_comparison( data1, data2, events = 5 ):
    dts = np.arange(0,data1.shape[1]*10,10)
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
            plt.xlabel(r'time ($\mu s$)',ha='right',x=1)
            plt.ylabel('amplitude',ha='right',y=1)
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
    plt.xlabel('area',ha='right',x=1)
    plt.ylabel('number of events',ha='right',y=1)


def gaussian(x, a, mu, sigma):
    return a * np.exp(-(x - mu)**2 / (2. * sigma**2))

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
    plt.xlabel(r'area ($\times 10^{-6}$)',ha='right',x=1)
    plt.ylabel('number of events (log)',ha='right',y=1)
    plt.legend()

    
def search_peaks(wf, n, ampllim = 9, plot = False): # search all the peaks of the signal
    # n: nb of bin-size for the dled delay (dled = "discret derivative")
    # ampllim: min amplitude to select a peak
    L = []
    t = np.arange(0,len(wf))
    dled = wf[:-n] - wf[n:] # derivative
    if plot:
        plt.plot(t[:-n], wf[:-n] - wf[n:], label = 'dled signal')
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
        plt.xlabel(r'time ($\mu s$)',ha='right',x=1)
        plt.ylabel('amplitude',ha='right',y=1)
        plt.legend()
    if inttot is not 0: return inttot
    

def integral_central_peak( wf, peaks_list, dtl = -2, dtr = 1,
                  tfit = 20, tlim = 100, tc = 5, tll = 5, tlr = 10, central=True, plot=False):
    # tlim = time window of the fit integration
    # tfit = time window of the fit
    # tc = min time to consider 2 peaks independently
    # tll = left limit of the window to search baseline value
    # tlr = right limit of the window to search Amin
    inttot = 0
    blmoy = wf.max()
    wf0 = wf - blmoy
    tt = np.array([i for i in range(len(wf))]) # time in bin-size
    tplot_tot = []
    A_tot = []
    for i in range(len(peaks_list)//2):
        dt = peaks_list[2*i + 1] - peaks_list[2*i]
        if dt < tc: continue #skip peaks if is too close to next one
        tlo = peaks_list[2*i]-tll
        if tlo < 0: tlo = 0
        bl = np.max(wf0[tlo:peaks_list[2*i]+1])
        Am = wf0[peaks_list[2*i]:peaks_list[2*i+1] + tlr]
        ttm = tt[peaks_list[2*i]:peaks_list[2*i+1] + tlr]
        Amin = np.min(Am) #local min of the signal
        tmin = ttm[Am == Amin][0] #time of the min
        if (central) and ((tmin-len(wf)/2 > 30) or (tmin-len(wf)/2 < -30)): continue #fit only central peak
        tl = tt[(tt <= tmin+dtr) & (tt >= tmin+dtl)]
        wfl = wf0[(tt <= tmin+dtr) & (tt >= tmin+dtl)]
        Il = -1*integ.simps(wfl, tl/100)
        # calculation of real integral of the waveform
        real_t = tt[(tt <= tmin+tlim) & (tt >= tmin+dtl)]
        real_wf = wf0[(tt <= tmin+tlim) & (tt >= tmin+dtl)]
        intreal = -1*integ.simps(real_wf, real_t/100)
        tr = tt[tmin+dtr:tmin+tfit] # time window for the fit
        amp = bl - Amin
        if len(tr) >= 5 and amp > 0:
            tr2 = tr - tr[0]
            #print('tmin:',tmin,'bl:',bl,'amp max:',amp)
            #try:
            fct_fit = expo2(bl) # fct used for the fit
            popt, pcov = curve_fit(fct_fit, tr2/100, wf0[tr],
                                    p0 = np.array([amp, 6.8]),
                                    bounds =  ([amp/1.5, 1], [amp*1.5, 10]))
            a, b = popt
            tnew = tt[tr[0]:]
            if plot:
                fct_fit_tot = fct_fit((tnew-tr[0])/100,a,b)
                tnew2 = min(tlim, tt[-1]-tr[0])
                tplot = tt[tmin+dtl-20:tmin+tnew2+20]
                plt.plot(tplot/100,wf0[tmin+dtl-20:tmin+tnew2+20],label='SiPM signal')
                plt.plot(tnew[:tnew2]/100, fct_fit_tot[:tnew2],
                         label=f'fit f(x) = baseline - a*exp(-b*x):\n a = {a:.2f}, b = {b:.2f}')
                plt.axhline(bl, color = 'r', label = 'baseline')
                plt.vlines((tmin+dtl)/100, wf0[tmin+dtl]-10, bl+10, colors = 'g',
                           label = 'integration limits')
                plt.vlines((tmin+dtr)/100, wf0[tmin+dtr]-10, bl+10, colors = 'g')
                plt.vlines((tmin+tfit)/100, wf0[tmin+tfit]-10, bl+10, colors = 'c',
                           label = 'fit limit')
                plt.vlines((tmin+tlim)/100, wf0[tmin+tlim]-10, bl+10, colors = 'g')
                plt.xlabel(r'time ($\mu s$)',ha='right',x=1)
                plt.ylabel('amplitude',ha='right',y=1)
                plt.legend()
                tlimplot = tr[0]
            fct_int = lambda x : bl - fct_fit(x, a, b)
            #print('bl, a, b : ', bl, a, b)
            Ir, err = integ.quad(fct_int, 0, (tlim-dtr)/100)
            inttot = Il + Ir
            #print(f'Integral: {Il:.1f} + {Ir:.1f} = {inttot:.2f}')
            tl = tt[tt >= peaks_list[2*i+1]+dtl]
    if inttot is not 0: return inttot


def spectrum_fit(peaks_integral,nsipm,volt,a=0,b=40,bins=1000,hlim=300,firstpe=1,lastpe=12, plot=False):
    h, t = np.histogram(peaks_integral, bins=bins, range=(a,b))
    pe = []
    pe_err = []
    # search peaks
    peaks, _ = find_peaks(h, height=hlim, width=5, distance=15)
    
    if plot:
        plt.figure(figsize=(12,6))
        plt.plot(t[:bins], h, '-', label = f'SiPM-{nsipm} at {volt} V')
        plt.plot(t[peaks], h[peaks], "x")
        plt.xlabel(r'area ($ADC\times \mu$s)',ha='right',x=1,fontsize=12)
        plt.ylabel('number of events',ha='right',y=1,fontsize=12)
    
    fit_not_failed = True
    npe = firstpe
    while fit_not_failed and npe <= lastpe and npe < len(peaks)+firstpe:
        i_max = peaks[npe-firstpe]
        t_max = t[i_max]
        di = 30
        t0, h0 = t[i_max-di:i_max+di], h[i_max-di:i_max+di]
        try:
            mu = t0[np.argmax(h0)]
            imax = np.argmax(h0)
            hmax = h0[imax]
            idx = np.where(h0>hmax/2) # fwhm 
            ilo, ihi = idx[0][0], idx[0][-1]
            sig = (t0[ihi]-t0[ilo]) / 2.355
            idx = np.where(((t0-mu) > -8 * sig) & ((t0-mu) < 8 * sig))
            idx0 = np.where(((t0-mu) > -4.5 * sig) & ((t0-mu) < 4.5 * sig))
            ilo, ihi = idx[0][0], idx[0][-1]
            ilo0, ihi0 = idx0[0][0], idx0[0][-1]
            t0, h0 = t0[ilo:ihi], h0[ilo:ihi]
            popt, pcov = curve_fit(gaussian, t0, h0, p0 = np.array([hmax, mu, sig]))
            perr = np.sqrt(np.diag(pcov))
            pe.append(popt[1])
            pe_err.append(perr[1])
            print(fr'PE {npe} at {popt[1]:.2f} +/- {perr[1]:.2f} ADC x us')
            if plot:
                X = np.linspace(a, b, num  = 100)
                plt.plot(X, gaussian(X, *popt), label = f'PE{npe} at {popt[1]:.2f} ADC x us')
                plt.legend(fontsize=12)
            npe += 1
        except:
            print(npe,'fit failed')
            fit_not_failed = False
            break
    return pe, pe_err


def retta(x, a, b):
    return a + b * x


def retta0(x, a):
    return a * x


def fit_pe(pe, pe_err, nsipm, volt, firstpe = 1, npe = 14, rlim = 0.1, offset = True):
    x = range(firstpe,len(pe)+firstpe)
    x0 = range(0,len(pe)+firstpe)
    func = retta0
    if offset: func = retta
    popt, pcov = curve_fit(func, x, pe)
    perr = np.sqrt(np.diag(pcov))
    
    # plot
    plt.figure(figsize=(12,6))
    plt.errorbar(x, pe, yerr=pe_err,color='b',marker='.',linestyle='',label=f'SiPM-{nsipm} at {volt} V')
    if offset: plt.plot(x0, func(x0, *popt), 'r-',
                        label=f'fit: $a+nPE\cdot b$ \n a=({popt[0]:.2f}$\pm${perr[0]:.2f}) $ADC~x~\mu$s \n b=({popt[1]:.2f}$\pm${perr[1]:.2f}) $ADC~x~\mu$s')
    else: plt.plot(x0, func(x0, *popt), 'r-',
                   label=f'fit: $nPE\cdot a$ \n a=({popt[0]:.2f}$\pm${perr[0]:.2f}) $ADC~x~\mu$s')
    plt.ylabel(r'area ($ADC\times \mu$s)',ha='right',y=1,fontsize=12)
    plt.xlabel('PE number',ha='right',x=1,size=40,fontsize=12)
    plt.legend(fontsize=12)
    plt.xlim(0,npe)
    
    # residuals
    residual = np.zeros(len(pe))
    for i,p in enumerate(pe):
        residual[i] = p-func(i+firstpe, *popt)
    plt.figure(figsize=(12,6))
    plt.errorbar(x,residual,yerr=pe_err,color='b',marker='.',linestyle='',label='residuals')
    plt.ylabel(r'area ($ADC\times \mu$s)',ha='right',y=1,fontsize=12)
    plt.xlabel('PE number',ha='right',x=1,size=40,fontsize=12)
    plt.axhline(0,color='r')
    plt.xlim(0,npe)
    plt.ylim(-rlim,rlim)
    plt.legend(fontsize=12)
    return popt, perr

def process_pe_spectrum(nsipm, v, firstpe, lastpe, maxarea, bins, hlim, xlim, ylim, date = 0):
    v_int = int(v)
    v_frac = int((v-v_int)*10)
    if date: peakint = np.load(f'SiPM{nsipm}/peakint_SiPM{nsipm}_{v_int}_{v_frac}_LED2p75_{date}.npy')
    else: peakint = np.load(f'SiPM{nsipm}/peakint_SiPM{nsipm}_{v_int}_{v_frac}_LED2p75.npy')
    npeaks = len(peakint[peakint>0])
    print('SiPM',nsipm,'with',v_int+v_frac/10,' V, Number of values:',npeaks)
    pe, pe_err = spectrum_fit(peakint,nsipm, v,1,maxarea,bins,hlim,firstpe,lastpe,plot=True)
    par, par_err = fit_pe(pe,pe_err,nsipm, v,firstpe,xlim,ylim,offset=1)
    return par, par_err, npeaks, pe, pe_err
