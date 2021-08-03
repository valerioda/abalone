import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.integrate as integ
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
import utility as psu
import process_utility as prsu
import time

def plot_data(data, fvolt=19, low=3.5, high=6, low_max= 1.5, high_max = 4, low_rt = 0, high_rt = 4,
              low_fom = 0, high_fom = 5, low_en = 2, high_en = 3.1):
    psu.plot_area_spectrum(data['area'], bins=200, logx=True, logy=False, volts=fvolt,low=low, high=high)
    psu.plot_area_spectrum(data['area_tot'], bins=200, logx=True, logy=False, volts=fvolt,low=low, high=high)
    #psu.plot_area_spectrum(data['area_filt'], bins=200, logx=True, logy=False, volts=fvolt,low=low, high=high)
    psu.plot_area_max(data['area'],data['peak_max'],bins=200,volts=fvolt,low=low,high=high,low2=low_max,high2=high_max)
    psu.plot_area_fom(data['area'],data['fom'],bins=200,volts=fvolt,low=low,high=high,low_fom=low_fom,high_fom=high_fom)
    #psu.plot_area_width(INTs,width50,bins=200)
    psu.plot_area_entropy(data['area'],data['entropy'],bins=200,volts=fvolt,low=low,high=high,low2=low_en,high2=high_en)
    psu.plot_area_risetime(data['area'],data['risetime'],bins=200,volts=fvolt,low=low,high=high,low2=low_rt,high2=high_rt)
    psu.plot_risetime_entropy(data['risetime'],data['entropy'],bins=200,volts=fvolt,low2=low_en,high2=high_en)
    
    
def process_abalone_data(filename,nn=0,nplot=5,width_calc=False,w=10):
    #fvolt=(filename.split('/')[-1].split('.')[0].split('_')[5])
    #fled=(filename.split('/')[-1].split('.')[0].split('_')[8])
    #sipmn=(filename.split('/')[-1].split('.')[0].split('_')[6])
    #sipmv=(filename.split('/')[-1].split('.')[0].split('_')[7])
    #fyear=(filename.split('/')[-1].split('.')[0].split('_')[0])
    #fmonth=(filename.split('/')[-1].split('.')[0].split('_')[1])
    #fday=(filename.split('/')[-1].split('.')[0].split('_')[2])
    #ftime=(filename.split('/')[-1].split('.')[0].split('_')[3])
    #print(f'Date {fyear}_{fmonth}_{fday} Time {ftime} ABALONE at {fvolt}')
    
    samples=1024
    data = np.fromfile(filename,  dtype=np.int16)
    n = len(data)/samples
    print(filename,'Number of events',n)
    data = np.array(np.array_split(np.array(data),n))
    
    print(f'Total events number: {len(data)}')
    MINs, FOMs, MAXs, INTs, width50, ENTRs, RTs, POSs, TOTs, FILs = [], [], [], [], [], [], [], [], [], []
    ts = time.time()
    plt.figure(figsize=(12,6))
    if nn == 0: nn = len(data)
    for i in range(nn):
        if (i % 100000) == 0: print(f'event n. {i} time to process: {time.time()-ts:.2f}')
        #bl = np.max(data[i][:])
        bl = np.mean(data[i][:40])
        wf = bl-data[i]
        ll, hh = int(len(wf)/2)-200, int(len(wf)/2)+200
        max_pos = np.where(wf==np.max(wf[ll:hh]))[0][0]
        POSs.append(np.where(wf==np.max(wf))[0][0])
        #datacut = data[np.array(max_pos)>1e6]
        area = np.sum(wf[max_pos-10:max_pos+90])
        if area > 1: fom = np.sum(wf[max_pos:max_pos+90])/area
        else: fom = 0
        MINs.append(bl)
        MAXs.append(np.max(wf))
        FOMs.append(fom)
        INTs.append(area/100)
        TOTs.append(np.sum(wf)/100)
        #wf_filt = gaussian_filter1d(wf, w)
        #FILs.append(np.sum(wf_filt)/100)
        if np.sum(wf) > 1:
            norm = np.abs(wf[wf!=0]/np.sum(wf))
            entropy = -np.sum(norm*np.log10(norm))
        else: entropy = 0
        ENTRs.append(entropy)
        try:
            t10 = np.where(wf[max_pos-10:hh]>=np.max(wf[ll:hh])*0.1)[0][0]
            t90 = np.where(wf[max_pos-10:hh]>=np.max(wf[ll:hh])*0.9)[0][0]
            risetime = t90-t10
        except:
            risetime = 0
        RTs.append(risetime)
        if width_calc & (np.max(wf)>20):
            ii, areafrac = 1, 0
            while areafrac < 0.5:
                ilo50, ihi50 = max_pos-ii, max_pos+ii
                areafrac = np.sum(wf[ilo50:ihi50])/area
                ii += 1
                wid50 = ihi50 - ilo50
            if i < nplot: plt.plot(wf,label=f'A={area:.1f}, width={wid50:.1f}')  
        else:
            wid50 = 0
            if i < nplot: plt.plot(wf, label=f'A={area:.1f}, entropy={entropy:.2f}, rt={risetime:.2f}')
        width50.append(wid50)

    plt.legend()
    data = pd.DataFrame(columns=['area','area_tot','area_filt','width','peak_max','max_pos','baselines','fom', 'entropy', 'risetime'])
    data['area'] = INTs
    data['area_tot'] = TOTs
    #data['area_filt'] = FILs
    if width_calc: data['width'] = width50
    data['peak_max'] = MAXs
    data['max_pos'] = POSs
    data['baselines'] = MINs
    data['fom'] = FOMs
    data['entropy'] = ENTRs
    data['risetime'] = RTs
    d_out = './processed_data'
    try: os.mkdir(d_out)
    except: pass
    #data.to_hdf(f'{d_out}/ABALONE_{fvolt}V_{sipmn}_{sipmv}_{fled}.h5', key='df', mode='w')
    #data.to_hdf(f'{d_out}/2021_07_20_T1637_ABALONE_19kV_SiPM2_30V_NoLED.h5', key='df', mode='w')
    return data


def select_data(data,filename,fvolt=18,entr_cut=20,max_cut=20,area_cut=(0,1e7),rt_cut=100,pos_cut=(100,900),
                events=10):
    wfs = psu.read_file(filename)
    
    # cut on area_up
    wfs_cut_areaup = wfs[(data['area']>area_cut[1])]
    try: psu.plot_waveforms( wfs_cut_areaup, events = events )
    except: pass
    print('Events rejected by areaup',len(wfs_cut_areaup),'->',len(wfs_cut_areaup)/len(data)*100,'%')
    print()
    #cut on area_down
    wfs_cut_areadown = wfs[(data['area']<area_cut[0])]
    try: psu.plot_waveforms( wfs_cut_areadown, events = events )
    except: pass
    print('Events rejected by areadown',len(wfs_cut_areadown),'->',len(wfs_cut_areadown)/len(data)*100,'%')
    print()
    # cut on entropy
    wfs_cut_entr = wfs[(data['entropy']>entr_cut)]
    try: psu.plot_waveforms( wfs_cut_entr, events = events )
    except: pass
    print('Events rejected by entropy',len(wfs_cut_entr),'->',len(wfs_cut_entr)/len(data)*100,'%')
    print()
    # cut on max
    wfs_cut_max = wfs[data['peak_max']<max_cut]
    try: psu.plot_waveforms( wfs_cut_max, events = events )
    except: pass
    print('Events rejected by max',len(wfs_cut_max),'->',len(wfs_cut_max)/len(data)*100,'%')
    print()
    # cut on risetime
    wfs_cut_rt = wfs[data['risetime']<rt_cut]
    try: psu.plot_waveforms( wfs_cut_rt, events = events )
    except: pass
    print('Events rejected by risetime',len(wfs_cut_rt),'->',len(wfs_cut_rt)/len(data)*100,'%')
    print()
    # cut on max position
    wfs_cut_pos = wfs[(data['max_pos']>pos_cut[1]) | (data['max_pos']<pos_cut[0])]
    try: psu.plot_waveforms( wfs_cut_pos, events = events )
    except: pass
    print('Events rejected by max position',len(wfs_cut_pos),'->',len(wfs_cut_pos)/len(data)*100,'%')
    print()
    # selection
    mask1 = (data['entropy']>0) & (data['entropy']<entr_cut) & (data['peak_max']>max_cut)
    mask2 = (data['area']<area_cut[1]) & (data['area']>area_cut[0]) & (data['risetime']>rt_cut)
    mask3 = (data['max_pos']<pos_cut[1]) & (data['max_pos']>pos_cut[0])
    mask = mask1 & mask2 & mask3
    wfs_sel = wfs[mask]
    data_sel = data[mask]
    try: psu.plot_waveforms( wfs_sel, events = events )
    except: pass
    #plot_data(data=data_sel,fvolt=fvolt,low=1.5, high=6.5,low3=1, high3=5)
    print('Events selected ',len(data_sel),'->',len(data_sel)/len(data)*100,'%')
    return data_sel, wfs_sel


def gauss(x,a,mu,sigma):
    return a*np.exp(-(x-mu)**2 / (2.*sigma**2))


def expo(x, a, b):
    return a*np.exp(-b*x)


def bimodal(x,a1,mu1,sigma1,a2,mu2,sigma2):#,a,b):
    return gauss(x,a1,mu1,sigma1)+gauss(x,a2,mu2,sigma2)#+expo(x,a,b)


def fit_pe_spectrum(area, bins = 200, volts = 10, low = 0, high = 100, fit_range=(0,100), dpe = False):
    area_space = np.linspace(low,high, bins)
    h, t = np.histogram(area, bins=area_space)
    plt.figure(figsize=(12,6))
    a1 = plt.hist(area,bins=area_space,histtype='step',lw=2,density=False)
    idx1, idx2 = np.where(t>fit_range[0])[0][0], np.where(t>fit_range[1])[0][0]
    imax = np.argmax(h[idx1:idx2])+idx1
    mu, hmax = t[imax], h[imax]
    idx = np.where(h[idx1:idx2]>hmax/2) # fwhm 
    ilo, ihi = idx[0][0], idx[0][-1]
    sig = (t[ihi]-t[ilo]) / 2.355
    
    if dpe:
        guess = (hmax, mu, sig, hmax/3, 2*mu, sig)
        bounds = ([hmax/2, mu-sig, 0, 0, 2*mu-sig, 0], [2*hmax, mu+sig, 2*sig, hmax, 2*mu+sig, 2*sig])
        popt, pcov = curve_fit(bimodal, t[idx1:idx2], h[idx1:idx2], p0 = guess, bounds = bounds)
    else:
        guess = (hmax, mu, sig)
        bounds = ([hmax/2, mu-sig, 0], [2*hmax, mu+sig, 2*sig])
        popt, pcov = curve_fit(gauss, t[idx1:idx2], h[idx1:idx2], p0 = guess, bounds = bounds)
    perr = np.sqrt(np.diag(pcov))
    if dpe: plt.plot(t, bimodal(t, *popt), label = f'PE fit')
    plt.plot(t, gauss(t, *popt[:3]), label = f'1PE = {popt[1]:.2f} $\pm$ {popt[2]:.2f} ADC x $\mu$s')
    #plt.plot(t, gauss(t, *guess[:3]), label = 'guess')
    if dpe: plt.plot(t, gauss(t, *popt[3:6]), label = f'2PE = {popt[4]:.2f} $\pm$ {popt[5]:.2f} ADC x $\mu$s')
    #plt.plot(t, expo(t, *popt[6:] ), label = f'exp = {popt[6]:.1f} + {popt[7]:.2f} x area')
    plt.title(f'ABALONE at {volts} kV')
    plt.xlabel('area (ADC x $\mu$s)',ha='right',x=1)
    plt.ylabel('counts',ha='right',y=1)
    plt.legend()
    return popt


def calculate_integrals( data, volts = 15, sipmv = 30, ledv = '3p0', nn = 0, ampllim = 5, tfit = 30, dtl=-10, dtr=3, plot = False, save = False):
    if nn == 0: nn = len(data)
    peakint = np.zeros(nn)
    print('Total events:',nn)
    t_start = time.time()
    for i in range(nn):
        if plot: plt.figure(figsize=(12,6))
        listpeaks = prsu.search_peaks(data[i], 4, ampllim=ampllim, plot = False)
        integral = prsu.integral_central_peak(data[i],listpeaks, dtl = dtl, dtr = dtr, tfit=tfit,
                                              tlim = 200, tc = 6, tll = 8, tlr = 10, plot = plot)
        if integral:
            peakint[i] = integral
        else:
            wf = np.mean(data[i][:40])-data[i]
            ll, hh = int(len(wf)/2)-200, int(len(wf)/2)+200
            max_pos = np.where(wf==np.max(wf[ll:hh]))[0][0]
            area = np.sum(wf[max_pos-10:max_pos+90])
            peakint[i] = area/100
        diff = time.time() - t_start
        if (i % 1000) == 0:
            print(f'event n. {i} area: {peakint[i]:.2f}, time to process: {diff:.2f}')
    if save: np.save(f'processed_data/peakint_ABALONE_{volts}kV_SiPM2_{sipmv}V_LED_{ledv}V.npy', peakint)
    return peakint