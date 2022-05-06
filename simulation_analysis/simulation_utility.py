import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import quad
import scipy.special as sc
from scipy.integrate import odeint
import scipy.integrate as integ
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import time
import json
import os

import math
from IPython.display import display, clear_output
import scipy.optimize as spo
import iminuit
import uproot


def read_waveforms_from_json(path = '/home/dandrea/abalone_simulation/results/SiPM', PE = 1, angle = 0, plot = False ):
    wfs = []
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
        bl = np.min(wf[tlo:peaks_list[2*i]+1])
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
            fct_int = lambda x : bl + fct_fit(x, a, b)
            #print('bl, a, b : ', bl, a, b)
            Ir, err = integ.quad(fct_int, 0, (tlim-dtr))
            #Ir, err = integ.quad(fct_int, 0, (tlim-dtr)/100)
            inttot = Il + Ir
            integrals.append(inttot)
            if plot:
                plt.figure(figsize=(8,4.5))
                fct_fit_tot = fct_fit((tnew-tr[0]),a,b)
                tnew2 = min(tlim, tt[-1]-tr[0])
                tplot = tt[tmin+dtl-20:tmin+tnew2+20]
                plt.plot(tplot,wf[tmin+dtl-20:tmin+tnew2+20],label=f'SiPM signal area= {inttot:.2f}')
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
            
            #print(f'Amp: {amp:.2f} Integral: {Il:.3f} + {Ir:.3f} = {inttot:.3f}')
            tl = tt[tt >= peaks_list[2*i+1]+dtl]
    return integrals

class track_and_readout:
    
    def __init__(self, pe, angle,file_number_range,main_path,save_path):
        self.angle = angle
        self.pe = pe
        self.file_number_range = file_number_range
        self.main_path = main_path
        self.save_path = save_path
        print("Initializing",self.pe,"pe, at angle",self.angle)
        
        self.area_collection = self.load_readout_data(self)
        self.track_data,self.total_events = self.load_track_data(self)
        self.e_stat,self.Edep_dist,self.e_ID = self.electron_count(self)
        
    def load_readout_data(self,arg):
        collection = []
        
        for num in self.file_number_range:
            path = self.main_path + "SiPM/"
            file_name = path + "SiPM_readout_" + str(self.pe) + "_" + str(self.angle) + "_run_" + str(num) +".txt"
            #print("reading " + file_name)
            #data = pd.read_json(file_name,lines=True)
            #data_column = np.array(data['voltages']) #.to_numpy()
            #for i in range(len(data_column)):
            #    collection.append(np.trapz(data_column[i]))
            with open(file_name) as f:
                for jsonObj in f:
                    data = json.loads(jsonObj)
                    try:
                        data_column = np.array(data['voltages'])
                        #for i in range(len(data_column)):
                        collection.append(np.trapz(data_column))
                    except: pass
        filename = (f'_{self.pe}PE_{self.angle}angle')
        path = self.save_path + "electron_counts/"
        if not os.path.exists(path):
            os.makedirs(path)
        np.save(path+'electron_count_area'+filename+'.npy',collection)
        return collection
    
    def load_track_data(self,arg):
        t_start = time.time()
        total_len = 0
        ID, Event, Parent, Track, Particle, X, Y, Z, Time, KE, DE, Volume = [], [], [], [],[], [], [], [],[], [], [], []
        df = pd.DataFrame(columns=["Event_ID","Parent_ID","Track_ID",
                                   "Particle","X","Y","Z","Time","KE","DE","Volume"])
        for fnum, num in enumerate(self.file_number_range):
            path = self.main_path + "tracking/"
            file_name = path + str(self.pe)+"_" + str(self.angle)+"_track_" + str(num) + ".root"
            file = uproot.open(file_name)
            tree = file["ntuple/ABALONE"]           
            
            Ev = tree.array("Event_ID")
            i_split = np.zeros(4,dtype=int)
            for ii in range(5):
                if ii != 0: i_split[ii-1] = np.where(Ev==ii)[0][0]
                Event.append(ii+total_len)
            Parent = np.append(Parent,np.split(tree.array("Parent_ID").astype(int),i_split))
            Track = np.append(Track,np.split(tree.array("Track_ID").astype(int),i_split))
            Particle = np.append(Particle,np.split(tree.array("Particle").astype(int),i_split))
            X = np.append(X,np.split(tree.array("X"),i_split))
            Y = np.append(Y,np.split(tree.array("Y"),i_split))
            Z = np.append(Z,np.split(tree.array("Z"),i_split))
            Time = np.append(Time,np.split(tree.array("Time"),i_split))
            KE = np.append(KE,np.split(tree.array("KE"),i_split))
            DE = np.append(DE,np.split(tree.array("DE"),i_split))
            Volume = np.append(Volume,np.split(tree.array("Volume").astype(int),i_split))
            total_len += 5
        df["Event_ID"], df["Parent_ID"] = Event, Parent
        df["Track_ID"], df["Particle"] = Track, Particle
        df["X"], df["Y"], df["Z"], df["Time"] = X, Y, Z, Time
        df["KE"], df["DE"] = KE, DE
        df["Volume"] = Volume
        #print(f'\ntime load with new method {time.time() - t_start:.2f}')
        #print('NEW DATAFRAME\n',df)
        return df,total_len
    
    def electron_count(self,arg): 
        print('counting electrons...')
        t_start = time.time()
        non_returning = 0     # electrons getting back-scattered to somewhere else
        non_returning_array = []

        returning_electron = 0     # electrons that end up in the scintillator
        returning_electron_array = []
        
        straight_electron = 0
        straight_electron_array = []
        
        never_electron = 0    # which never goes into the scintillator
        never_array = [] 
        
        non_returning_event=[]
        returning_electron_event=[]
        straight_event = []
        never_event=[]
        
        sc_vol = 15
        for i in range(0,len(self.track_data["Event_ID"])):
            prt = self.track_data['Parent_ID'][i]
            trk = self.track_data['Track_ID'][i]
            vol = self.track_data['Volume'][i][prt==0]
            for j in list(set(trk[prt==0])):
                energy = self.track_data["DE"][i][(prt==0) & (trk==j)].sum()
                if sc_vol in vol and vol[-1] != sc_vol:
                    non_returning += 1
                    non_returning_array.append(energy)
                    non_returning_event.append(i)
                if sc_vol in vol and vol[-1] == sc_vol:
                    idx = np.array(np.where(vol==sc_vol)[0])
                    maxi, mini = np.max(idx), np.min(idx)
                    if np.sum(idx) == (len(idx)/2)*(mini + maxi): flag = 0
                    else: flag = 1
                    if flag == 0:
                        straight_electron += 1
                        straight_electron_array.append(energy)
                        straight_event.append(i)
                    else:
                        returning_electron += 1
                        returning_electron_array.append(energy)
                        returning_electron_event.append(i)
                if not sc_vol in vol:
                    never_electron +=1
                    never_array.append(energy)
                    never_event.append(i)
        
        print(f'time to count electrons {time.time() - t_start:.2f}')
        print('straight event', len(straight_event))
        print('non returning event', len(non_returning_event))
        print('returning electron event', len(returning_electron_event))
        print('never event', len(never_event))
        elcount = np.array([straight_electron,returning_electron,non_returning,never_electron])
        elcount_event = [straight_event,returning_electron_event,non_returning_event,never_event]
        elcount_energy = [straight_electron_array,returning_electron_array,non_returning_array,never_array]
        path = self.save_path + "electron_counts/"
        if not os.path.exists(path):
            os.makedirs(path)
        filename = (f'_{self.pe}PE_{self.angle}angle')
        np.save(path+'electron_count'+filename+'.npy',elcount)
        np.save(path+'electron_count_event'+filename+'.npy',elcount_event)
        np.save(path+'electron_count_energy'+filename+'.npy',elcount_energy)
        return elcount, elcount_event, elcount_energy


def gaussian(x, a,mu,sig):
    return a/(math.sqrt(2.*math.pi)*sig)*np.exp(-np.power((x - mu)/sig, 2.)/2)

def leastsquare(a,mu,sig):  #parameter: mu,sigma and (nu?! is nu = mu?!)
    return sum((gaussian(x,a,mu,sig)-y)**2 for x, y in zip(chosen_bin, chosen_yhist))

def plot_spectra(path,PE = 1, angle = 0, voltage = 25, xlim = (200,12000), bin_number = 150):
    print(f'Results with {PE} PE at angle {angle} deg')
    
    stat = np.load(path+f'electron_counts/electron_count_{PE}PE_{angle}angle.npy')
    e_ID = np.load(path+f'electron_counts/electron_count_event_{PE}PE_{angle}angle.npy')
    area_collection = np.load(path+f'electron_counts/electron_count_area_{PE}PE_{angle}angle.npy')
    Edep_dist = np.load(path+f'electron_counts/electron_count_energy_{PE}PE_{angle}angle.npy')
    
    area_straight_list = np.array([area_collection[i] for i in e_ID[0]])
    area_returning_list = np.array([area_collection[i] for i in e_ID[1]])
    area_non_list = np.array([area_collection[i] for i in e_ID[2]])
    
    yhist,binedges = np.histogram(area_straight_list,bins=bin_number)
    yhistr,binedgesr = np.histogram(area_returning_list,bins=bin_number)
    yhistn,binedgesn = np.histogram(area_non_list,bins=bin_number)
    bc = np.array((binedges[1:] + binedges[:-1])/2)
    
    #plt.figure(figsize=(8,4.5))
    #plt.plot(binedges[1:],yhist,label='straight')
    #plt.plot(binedges[1:],yhistr,label='returning')
    #plt.plot(binedges[1:],yhistn,label='non-returning')
    #plt.legend()
    
    #peak_index = np.argmax(yhist)
    #left = 30
    #right = len(yhist)-peak_index - 1

    #global chosen_bin
    #global chosen_yhist

    #chosen_bin = bc[peak_index-left:peak_index+right]
    #chosen_yhist= yhist[peak_index-left:peak_index+right]
    
    #m = iminuit.Minuit(leastsquare,a=max(yhist),mu=bc[peak_index],sig=500,error_a=100, error_mu=100,error_sig=500,errordef=0.5)
    #m = iminuit.Minuit(leastsquare,a=max(yhist),mu=bc[peak_index],sig=2000)
    #m.migrad()
    #m.hesse()
    #print(m.values)
    ##print(m.errors)

    x= np.linspace(xlim[0],xlim[1],bin_number)
    #y = gaussian(x,m.values[0],m.values[1],m.values[2])
    
    #popt, pcov = curve_fit(gaussian, chosen_bin, chosen_yhist)
    #                       p0=(max(yhist), bc[peak_index],1000))
    
    popt, pcov = curve_fit(gaussian, bc, yhist, p0=(np.max(yhist), bc[np.argmax(yhist)],1000))
    perr = np.sqrt(np.diag(pcov))
    y = gaussian(x, *popt)
    plt.figure(figsize=(8,4.5))
    plt.scatter(bc,yhist,label = "Straight electrons distribution")
    #plt.scatter(chosen_bin,chosen_yhist,color='r',label="Fitting Data Points")
    plt.plot(x,y,label='Gaussian Fit',color = 'black')
    #plt.axvline(bc[peak_index-left],color='red',linestyle=":")
    #plt.axvline(bc[peak_index+right],color='red',linestyle=":",label = "Fitting Range")
    plt.ylim(0)
    plt.legend(prop={'size': 8})
    plt.show()
    
    print(" ** Summary of the electrons ** ")
    print(f'Total electrons: {np.sum(stat)}')
    print(f'Portion of straight electrons: {(stat[0])/np.sum(stat):.2f}')
    print(f'Portion of returning back-scattered electrons: {(stat[1])/np.sum(stat):.2f}')
    print(f'Portion of non-returning back-scattered electrons: {(stat[2])/np.sum(stat):.2f}')
    print(f'Portion of back-scattered electrons (returning+non-returning): {(stat[1]+stat[2])/np.sum(stat):.2f}')
    print(f'Portion of undetected electrons: {(stat[3])/np.sum(stat):.2f}')

    ########################
    ########################
    path = './plots/'
    plt.figure(figsize=(8,4.5))
    plt.hist(area_straight_list,bins=bin_number,label = "Straight",histtype='step',color='g') #,normed=True
    plt.hist(area_returning_list,bins=bin_number,label = "Returning",histtype='step',color='b') #,normed=True
    plt.hist(area_non_list,bins=bin_number,label = "Non-Returning",histtype='step',color='r') #,log=True   ,normed=True
    plt.plot(x,y,label="Gaussian Fit",color = 'black')
    #plt.set_title("SiPM area count",fontsize=11)
    plt.xlabel('area [a.u.]',ha='right',x=1,fontsize=12)
    plt.ylabel('counts',ha='right',y=1,fontsize=12)
    plt.tick_params(axis='x',labelsize=12)
    plt.tick_params(axis='y',labelsize=12)
    plt.legend(fontsize=12)
    plt.savefig(path+"Spectra_"+str(PE)+"_pe_angle_"+str(angle)+"_voltage"+str(voltage),dpi=800)
    
    fig, axs = plt.subplots(2,2,figsize=(13,8))
    axs[0,0].hist(area_straight_list,bins=bin_number,label = "Straight",histtype='step',color='g') #,normed=True
    axs[0,0].hist(area_returning_list,bins=bin_number,label = "Returning",histtype='step',color='b') #,normed=True
    axs[0,0].hist(area_non_list,bins=bin_number,label = "Non-Returning",histtype='step',color='r') #,log=True   ,normed=True
    axs[0,0].plot(x,y,label="Gaussian Fit",color = 'black')
    axs[0,0].set_title("SiPM area count",fontsize=11)
    axs[0,0].legend(prop={'size': 8})
    axs[0,0].set(xlabel="Area")
    axs[0,0].relim()
    axs[0,0].autoscale_view()

    axs[1,0].hist(Edep_dist[0],bins=bin_number,histtype='step',label = "Straight e-",color='g') #,log=True
    axs[1,0].hist(Edep_dist[1],bins=bin_number,histtype='step',label = "Returning e-",color='b') #,log=True
    axs[1,0].hist(Edep_dist[2],bins=bin_number,histtype='step',label = "Non-retrurning",color='r')
    axs[1,0].set_title("Energy deposited",fontsize=11)
    axs[1,0].set(xlabel="Energy (keV)")
    axs[1,0].legend(prop={'size': 8})

    axs[0,1].hist(area_straight_list,bins=bin_number,label = "Straight e-",histtype='step',color='g',log=True) #,normed=True
    axs[0,1].hist(area_returning_list,bins=bin_number,label = "Returning e-",histtype='step',color='b',log=True) #,normed=True
    axs[0,1].hist(area_non_list,bins=bin_number,label = "Non-returning",histtype='step',color='r',log=True) #,log=True   ,normed=True
    axs[0,1].set_title("SiPM area count (log-scale)",fontsize=11)
    axs[0,1].legend(prop={'size': 8})
    axs[0,1].set(xlabel='Area')

    axs[1,1].hist(Edep_dist[0],bins=bin_number,histtype='step',color='g',label = "Straight e-",log=True) #,log=True
    axs[1,1].hist(Edep_dist[1],bins=bin_number,histtype='step',color='b',label = "Returning e-",log=True) #,log=True
    axs[1,1].hist(Edep_dist[2],bins=bin_number,histtype='step',color='r',label = "Non-returning",log=True) #,log=True
    axs[1,1].legend(prop={'size': 8})
    axs[1,1].set_title("Energy depposited (log-scale)",fontsize=11)
    axs[1,1].set(xlabel="Energy (keV)")

    fig.suptitle(str(PE)+ " PE event Spectra at angle "+str(angle)+" deg",fontsize=12)
    plt.subplots_adjust(hspace = 0.25)
    
    if not os.path.exists(path):
        os.makedirs(path)
        
    plt.savefig(path+"SpectraAll_"+str(PE)+"_pe_angle_"+str(angle)) 
    plt.show()
    
    return popt
    
def e_stat_bootstrap(path,PE=1,angle=0):
    e_stat = np.load(path+f'electron_counts/electron_count_{PE}PE_{angle}angle.npy')
    e_ID = np.load(path+f'electron_counts/electron_count_event_{PE}PE_{angle}angle.npy')
    area_collection = np.load(path+f'electron_counts/electron_count_area_{PE}PE_{angle}angle.npy')
    Edep_dist = np.load(path+f'electron_counts/electron_count_energy_{PE}PE_{angle}angle.npy')
    
    s = np.ones(e_stat[0])
    r = np.ones(e_stat[1])*2
    nr = np.ones(e_stat[2])*3
    ne = np.ones(e_stat[3])*4
    sample = np.concatenate([s, r,nr,ne], axis=None)
    
    bootstrap_trials = 10000
    bootstrap_nr_portion = []
    bootstrap_s_portion = []
    bootstrap_r_portion = []
    bootstrap_ne_portion= []
    
    for x in range(0, bootstrap_trials):
        bootstrap_data = np.random.choice(sample, 8000, replace=True)
        # count the number of non-returning back-scattered e-
        bootstrap_nr_stat = np.sum(bootstrap_data == 3)/len(bootstrap_data)
        bootstrap_s_stat = np.sum(bootstrap_data == 1)/len(bootstrap_data)
        bootstrap_r_stat = np.sum(bootstrap_data == 2)/len(bootstrap_data)
        bootstrap_ne_stat = np.sum(bootstrap_data == 4)/len(bootstrap_data)

        bootstrap_nr_portion += [np.average(bootstrap_nr_stat)]
        bootstrap_s_portion += [np.average(bootstrap_s_stat)]
        bootstrap_r_portion += [np.average(bootstrap_r_stat)]
        bootstrap_ne_portion += [np.average(bootstrap_ne_stat)]
        
    bootstrap_nr_sorted = np.sort(bootstrap_nr_portion)
    bootstrap_ne_sorted = np.sort(bootstrap_ne_portion)
    bootstrap_s_sorted = np.sort(bootstrap_s_portion)
    bootstrap_r_sorted = np.sort(bootstrap_r_portion)
    plt.figure(figsize=(8,4.5))
    plt.hist(bootstrap_nr_portion,bins=25)
    plt.axvline(bootstrap_nr_sorted[int(bootstrap_trials*0.05)],color='r')
    plt.axvline(bootstrap_nr_sorted[int(bootstrap_trials*0.95)],color='r')
    plt.title("Bootstrapping of the portion of NRBE at " + str(PE) + " pe angle "+str(angle)+" deg",fontsize=11)
    path = "./plots/NRBE/"
    if not os.path.exists(path):
        os.makedirs(path)
        
    plt.savefig(path+str(PE)+"_pe_angle_"+str(angle))
    plt.show()
    
    nr_stat = [np.median(bootstrap_nr_portion),bootstrap_nr_sorted[int(bootstrap_trials*0.05)],\
               bootstrap_nr_sorted[int(bootstrap_trials*0.95)],np.std(bootstrap_nr_portion)]
    ne_stat = [np.median(bootstrap_ne_portion),bootstrap_ne_sorted[int(bootstrap_trials*0.05)],\
               bootstrap_ne_sorted[int(bootstrap_trials*0.95)],np.std(bootstrap_ne_portion)]
    s_stat = [np.median(bootstrap_s_portion),bootstrap_s_sorted[int(bootstrap_trials*0.05)],\
               bootstrap_s_sorted[int(bootstrap_trials*0.95)],np.std(bootstrap_s_portion)]
    r_stat = [np.median(bootstrap_r_portion),bootstrap_r_sorted[int(bootstrap_trials*0.05)],\
               bootstrap_r_sorted[int(bootstrap_trials*0.95)],np.std(bootstrap_r_portion)]
    return [s_stat,r_stat,nr_stat,ne_stat]



def calculate_integrals(data, nn = 0, inf = 480, sup = 550, calc_int = False,
                        plot = False, save = False, info = 'LYSO_background' ):
    if nn == 0: nn = len(data)
    peakint = np.zeros(nn)
    print('Total events:',nn)
    MAXs, AREAs, POSs, TAUs, INTs = [], [], [], [], []
    t_start = time.time()
    wsize = data.shape[1]
    hsize= int(wsize/2)
    tt = np.array([i for i in range(wsize)]) # time in bin-size
    for i in range(nn):
        diff = time.time() - t_start
        if ((i+1) % 5000) == 0:
            print(f'event n. {i+1}, time to process: {diff:.2f}')
        
        bl = np.mean(data[i][hsize-100:hsize-30])
        wf = data[i]-bl
        #wf = bl - data[i]
        max_pos = np.where(wf==np.max(wf))[0][0]
        maxx = np.max(wf)
        area = np.sum(wf)
        try:
            tt10 = np.where(wf[max_pos:]<maxx*0.1)[0][0] + max_pos
            tt90 = np.where(wf[max_pos:]<maxx*0.9)[0][0] + max_pos
            tau = tt10 - tt90
        except:
            tau = 0
        
        # INTEGRAL CALCULATION
        ww, hh = 8, 14
        dled = wf[ww:] - wf[:-ww]
        listpeaks,_ = find_peaks(dled, height=hh,distance=10)
        peakpos = listpeaks[(listpeaks < sup) & (listpeaks > inf)]
        if (len(peakpos) != 1) or (maxx <= 0) or (calc_int == False):
            if plot & calc_int:
                print('peak position not found in trigger region',listpeaks)
                plt.figure(figsize=(12,6))
                plt.plot(tt,wf,label='signal')
                plt.plot(tt[:-ww],dled,label='derivative')
                #plt.xlim(inf,sup)
                plt.legend()
            INTs.append(0)
            POSs.append(0)
        else:
            peakpos = peakpos[0]
            # fit parameters ----- could be needed to improve them
            dtl, dtr, tfit, tlim, tll = -15, 110, 350, 600, 30
            tl = tt[(tt <= peakpos+dtr) & (tt >= peakpos+dtl)]
            wfl = wf[(tt <= peakpos+dtr) & (tt >= peakpos+dtl)]
            Il = integ.simps(wfl, tl)
            area = np.sum(wf[peakpos+dtl:peakpos+tlim])
            # calculation of real integral of the waveform
            try:
                real_t = tt[(tt <= peakpos+tlim) & (tt >= peakpos+dtl)]
                real_wf = wf[(tt <= peakpos+tlim) & (tt >= peakpos+dtl)]
                intreal = integ.simps(real_wf, real_t)
                tr = tt[peakpos+dtr:peakpos+tfit] # time window for the fit
                bl = np.mean(wf[hsize-tll:hsize+dtl])
                tr2 = tr - tr[0]
                def expo(c):
                    return(lambda x, a, b : c + a*np.exp(-b*x))
                fct_fit = expo(bl) # fct used for the fit
                guess = [maxx, 0.07]
                popt, pcov = curve_fit(fct_fit, tr2, wf[tr],p0 = guess,
                                    bounds =  ([maxx*0.1, 0.001], [maxx*1.5, 0.2]))
                a, b = popt
                tnew = tt[tr[0]:]
                fct_fit_tot = fct_fit((tnew-tr[0]),a,b)
                fct_fit_g = fct_fit((tnew-tr[0]),guess[0],guess[1])
                tnew2 = min(tlim, tt[-1]-tr[0])
                tplot = tt[peakpos + dtl -20 : peakpos + tnew2 + 20]
                if plot:
                    plt.figure(figsize=(12,6))
                    plt.plot(tplot,wf[peakpos+dtl-20:peakpos+tnew2+20],label='signal')
                    plt.plot(tnew[:tnew2], fct_fit_tot[:tnew2],
                            label=f'fit f(x) = bl + a*exp(-b*x):\n a = {a:.2f}, b = {b:.4f}')
                    plt.axhline(bl, color = 'r', label = 'baseline')
                    plt.vlines((peakpos+dtl), wf[peakpos+dtl]-10, bl+10, colors = 'g',
                                label = 'integration limits')
                    plt.vlines((peakpos+dtr), wf[peakpos+dtr]-10, bl+10, colors = 'g')
                    plt.vlines((peakpos+tfit), wf[peakpos+tfit]-10, bl+10, colors = 'c',
                                label = 'fit limit')
                    plt.vlines((peakpos+tlim), wf[peakpos+tlim]-10, bl+10, colors = 'g')
                    plt.xlabel(r'time (samples)',ha='right',x=1)
                    plt.ylabel('amplitude',ha='right',y=1)
                    plt.legend()
                fct_int = lambda x : bl + fct_fit(x, a, b)
                Ir, err = integ.quad(fct_int, 0, (tlim-dtr))
                inttot = Il + Ir
                INTs.append(inttot)
                POSs.append(peakpos)
            except:
                INTs.append(0)
                POSs.append(0)
        AREAs.append(area)
        TAUs.append(tau)
        MAXs.append(np.max(wf))
    data = pd.DataFrame(columns=['area','integral','peak_max','max_pos','tau'])
    data['area'] = AREAs
    data['integral'] = INTs
    data['peak_max'] = MAXs
    data['max_pos'] = POSs
    data['tau'] = TAUs
    if save: data.to_hdf(f'processed_data/data_tailFit_ABALONE_{volts}kV_SiPM2_{sipmv}V_{info}.h5', key='df', mode='w')
    return data