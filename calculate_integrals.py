import os
import argparse
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from scipy.integrate import quad
import scipy.special as sc
from scipy.integrate import odeint
import process_utility as psu
import scipy.integrate as integ
from scipy.optimize import curve_fit
import time

from sklearn.externals.joblib import Parallel, delayed
import multiprocessing
num_cores = multiprocessing.cpu_count()


def main():
    
    par = argparse.ArgumentParser(description = 'SiPM integral calculate')
    arg, st, sf = par.add_argument, 'store_true', 'store_false'
    arg('-d', '--date', nargs=1, action='store', help = 'Date in format YYYY_MM_DD')
    arg('-s', '--sipm', nargs=1, action='store', help = 'SiPM number')
    arg('-n', '--number', nargs=1, action='store', help = 'Number of peak to integrate')
    arg('-sc', '--sipmcalibration', nargs=1, action='store', help = 'SiPM Calibration')
    arg('-v', '--voltages', nargs=2, action='store', help = 'Voltage interval')
    arg('-f', '--file', nargs=1, action='store', help = 'Data file')
    arg('-ft', '--fitparameters', nargs=5, action='store', help = 'Fit parameters')
    arg('-st', '--searchparameters', nargs=2, action='store', help = 'Search parameters')
    arg('-p', '--parallelize', action=st, help = 'Parallelize integrals')
    args = vars(par.parse_args())

    if args['date']:
        date = args['date'][0]
    else:
        print('Date not given [format YYYY_MM_DD]')
        return
    
    if args['sipm']:
        sipm = args['sipm'][0]
    else:
        print('Number of SiPM not given')
        return
    
    parameters = [-4, 1, 10, 100, 4]
    if args['fitparameters']:
        parameters = np.array(args['fitparameters'][:],dtype=int)
    searchparameters = [4, 2]
    if args['searchparameters']:
        searchparameters = np.array(args['searchparameters'][:],dtype=int)
    
    npeaks = 0
    if args['number']: npeaks = int(args['number'][0])
    local_dir = '.'
    
    # START SIPM CALIBRATION
    if args['sipmcalibration']:
        
        if args['voltages']:
            vstart, vend = float(args['voltages'][0]), float(args['voltages'][1])+0.1
        else:
            print('Voltage interval not given')
        return
        step = 0.5
        d_out = f'{local_dir}/SiPM{sipm}'
        try: os.mkdir(d_out)
        except: pass
        
        print('Calculate integrals for SiPM n.',sipm,', voltage from ',vstart,'to',vend,'V')
    
        voltages = np.arange(vstart,vend,step)
        for v in voltages:
            v_int = int(v)
            v_frac = int((v-v_int)*10)
            try: data = psu.read_file(f'/data/abalone/{date}/SiPM{sipm}_{v_int}_{v_frac}_LED2p75.dat')
            except: return
            print('SiPM',sipm,'Voltage =',v_int+v_frac/10,'V, Total events:',len(data))
            peakint = calculate_integrals( data, sipm, searchparameters, parameters, npeaks, d_out )
            np.save(f'{d_out}/peakint_SiPM{sipm}_{v_int}_{v_frac}_LED2p75_{date}.npy', peakint)
            return

    # NORMAL ANALYSIS
    if args['file']:
        datafile = args['file'][0]
    else:
        print('Data file not given')
        return
    d_out = f'{local_dir}/{date}'
    try: os.mkdir(d_out)
    except: pass
    try: data = psu.read_file(f'{datafile}')
    except:
        print('File not valid')
        return
    print('Calculate integrals for ',datafile,'Total events:',len(data))
    dataname = datafile.split('/')[-1].split('.')[0]
    if args['parallelize']:
        peakint = calculate_integrals_par( data, sipm, searchparameters, parameters, npeaks, d_out )
    else:
        peakint = calculate_integrals( data, sipm, searchparameters, parameters, npeaks, d_out )
    np.save(f'{d_out}/peakint_{dataname}.npy', peakint)
    return
    
    
def calculate_integrals( data, sipm, spar, par, npeaks = 0, d_out = '.' ):
    print('Use of standard loop')
    nn = len(data)
    peakint = np.zeros(nn)
    t_start = time.time()
    print('Search parameters:',spar)
    print('Fit parameters:',par)
    if npeaks == 0: npeaks = nn
    for i in range(npeaks):
        listpeaks = psu.search_peaks(data[i], spar[0], spar[1], False)
        peakint[i] = psu.integral_central_peak(data[i],listpeaks,par[0],par[1],par[2],par[3],par[4],8,10,central=False)
        diff = time.time() - t_start
        if (i % 1000) == 0:
            print(f'event n. {i} area: {peakint[i]:.2f}, time to process: {diff:.2f}')
    print(f'total time to process: {diff:.2f}')
    print()
    return peakint


def parrallel_integrals(wf,spar,par):
    listpeaks = psu.search_peaks(wf, spar[0], spar[1], False)
    peakint = psu.integral_central_peak(wf,listpeaks,par[0],par[1],par[2],par[3],par[4],8,10,central=False)
    return peakint

def calculate_integrals_par( data, sipm, spar, par, npeaks = 0, d_out = '.' ):
    print('Use of parallelized loop')
    nn = len(data)
    peakint = np.zeros(nn)
    t_start = time.time()
    print('Search parameters:',spar)
    print('Fit parameters:',par)
    if npeaks == 0: npeaks = nn
    peakint = Parallel(n_jobs=num_cores)(delayed(parrallel_integrals)(data[i],spar,par) for i in range(npeaks)  )
    peakint = np.array(peakint)
    
    diff = time.time() - t_start
    print(f'Total time to process: {diff:.2f}')
    return peakint

if __name__=="__main__":
    main()
