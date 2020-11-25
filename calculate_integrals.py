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

def main():
    
    par = argparse.ArgumentParser(description = 'SiPM integral calculate')
    arg, st, sf = par.add_argument, 'store_true', 'store_false'
    arg('-d', '--date', nargs=1, action='store', help = 'Date in format YYYY_MM_DD')
    arg('-s', '--sipm', nargs=1, action='store', help = 'SiPM number')
    arg('-v', '--voltages', nargs=2, action='store', help = 'Voltage interval')
    arg('-st', '--step', nargs=1, action='store', help = 'Voltage step')
    arg('-n', '--number', nargs=1, action='store', help = 'Number of peak to integrate')
    args = vars(par.parse_args())
    
    if args['date']:
        date = args['date'][0]
    else:
        print('Date not given [format YYYY_MM_DD]')
        return
    
    if args['sipm']:
        sipm = args['sipm'][0]
    else:
        #print('Number of SiPM not given')
        return
    
    if args['voltages']:
        vstart, vend = float(args['voltages'][0]), float(args['voltages'][1])+0.1
    else:
        print('Voltage interval not given')
        return
    
    step = 0.5
    if args['step']: step = float(args['step'][0])
    
    npeaks = 0
    if args['number']: npeaks = int(args['number'][0])

    local_dir = '.'
    d_out = f'{local_dir}/SiPM{sipm}'
    try: os.mkdir(d_out)
    except: pass

    print('Calculate integrals for SiPM n.',sipm,', voltage from ',vstart,'to',vend,'V')
    
    #integrals = []
    voltages = np.arange(vstart,vend,step)
    for v in voltages:
        v_int = int(v)
        v_frac = int((v-v_int)*10)
        peakint = calculate_integrals(sipm,date,v_int,v_frac,npeaks,d_out)
        #integrals.append(peakint)
    return

def calculate_integrals( sipm, date, v_int, v_frac, npeaks = 0, d_out = '.' ):
    try: data = psu.read_file(f'/data/abalone/{date}/SiPM{sipm}_{v_int}_{v_frac}_LED2p75.dat')
    except: return
    nn = len(data)
    peakint = np.zeros(nn)
    print('SiPM',sipm,'Voltage =',v_int+v_frac/10,'V, Total events:',nn)
    t_start = time.time()
    if npeaks == 0: npeaks = nn
    for i in range(npeaks):
        listpeaks = psu.search_peaks(data[i], 4, 2, False)
        peakint[i] = psu.integral_central_peak(data[i],listpeaks,-5,1,10,100,5,8,10)
        diff = time.time() - t_start
        if (i % 1000) == 0:
            print(f'event n. {i} area: {peakint[i]:.2f}, time to process: {diff:.2f}')
    np.save(f'{d_out}/peakint_SiPM{sipm}_{v_int}_{v_frac}_LED2p75_{date}.npy', peakint)
    print()
    return peakint

if __name__=="__main__":
    main()
