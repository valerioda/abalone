import argparse
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import os
from IPython.display import display, clear_output
import scipy.optimize as spo
import iminuit
import uproot
import json
import multiprocessing as mp
import time
from scipy.optimize import curve_fit


def main():
    par = argparse.ArgumentParser(description = 'ABALONE track readout analysis')
    arg, st, sf = par.add_argument, 'store_true', 'store_false'
    arg('-a', '--angle', nargs=1, action='store', help = 'angle')
    arg('-p', '--pe', nargs=1, action='store', help = 'PE number')
    arg('-fn', '--filenumber', nargs=1, action='store', help = 'file number')
    args = vars(par.parse_args())

    if args['angle']: angle = int(args['angle'][0])
    else: angle = 0

    if args['pe']: PE = int(args['pe'][0])
    else: PE = 1

    if args['filenumber']: fn = int(args['filenumber'][0])
    else: fn = 2000
    
    print('Track and readout analysis with',PE,'PE and',angle,'angle')
    file_number_range = range(1,fn)
    main_path = '/home/pieramico/AIUTO/provo/'
    save_path = './'
    #/home/dandrea/abalone_simulation/results/nrbe/'
    pe1_ang0 = track_and_readout(PE,angle,file_number_range,main_path)
    plot_spectra(pe1_ang0,save_path)
    return


class track_and_readout:
    
    def __init__(self, pe, angle,file_number_range,main_path):
        self.angle = angle
        self.pe = pe
        self.file_number_range = file_number_range
        self.main_path = main_path
        print("Initializing",self.pe,"pe, at angle",self.angle)
        
        self.area_collection = self.load_readout_data(self)
        self.track_data,self.total_events = self.load_track_data(self)       
        self.e_stat,self.Edep_dist,self.e_ID = self.electron_count(self)
        
    def load_readout_data(self,arg):
        collection = []
        
        for num in self.file_number_range:
            clear_output(wait=True)
            path = self.main_path + "SiPM/"
            file_name = path + "SiPM_readout_" + str(self.pe) + "_" + str(self.angle) + "_run_" + str(num) +".txt"
            print("reading " + file_name)
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
                        
        return collection

    def load_track_data(self,arg):
        all_loaded_data = pd.DataFrame(columns=["Event_ID","Parent_ID","Track_ID","Particle","X","Y","Z","Time","KE","DE","Volume"])
        total_len = 0
        
        for num in self.file_number_range:
            
            clear_output(wait=True)
            
            path = self.main_path + "tracking/"
            file_name = path + str(self.pe)+"_" + str(self.angle)+"_track_" + str(num) + ".root"
            
            print("reading " + file_name)#display("reading " + os.path.join(r,filename))
            
            file = uproot.open(file_name)
            tree = file["ntuple/ABALONE"]
            df = pd.DataFrame(columns=["Event_ID","Parent_ID","Track_ID","Particle","X","Y","Z","Time","KE","DE","Volume"])
            df["Event_ID"] = (tree.array("Event_ID")).astype(int)+int(total_len)
            df["Parent_ID"] = (tree.array("Parent_ID")).astype(int)
            df["Track_ID"] = (tree.array("Track_ID")).astype(int)
            df["Particle"] = tree.array("Particle")
            df["X"] = tree.array("X")
            df["Y"] = tree.array("Y")
            df["Z"] = tree.array("Z")
            df["Time"] = tree.array("Time")
            df["KE"] = tree.array("KE")
            df["DE"] = tree.array("DE")
            df["Volume"] = (tree.array("Volume")).astype(int)

            df.loc[df["Particle"] == 0 , "Particle"] = "e-"
            df.loc[df["Particle"] == 1 , "Particle"] = "photon"

            frames = [all_loaded_data,df]
            all_loaded_data = pd.concat(frames)
            total_len += len(set(df["Event_ID"].values))

        all_loaded_data["Event_ID"] = pd.to_numeric(all_loaded_data["Event_ID"], downcast='signed')
        return all_loaded_data,total_len
    
   
    def electron_count(self,arg):  #counting non-returning backscattered electron directly, and their energy deposited
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
                
        #display(len(non_returning_array)+len(good_electron_array)+len(never_array))
        sc_vol = 15
        for i in range(0,int(max(self.track_data["Event_ID"])+1)):
            event_select = self.track_data['Event_ID'] == i
            electron_select = self.track_data['Parent_ID']==0
            scintillator = self.track_data['Volume'] == sc_vol
            temp = self.track_data[event_select & electron_select]

            for j in range (int(min(temp["Track_ID"])),int(max(temp["Track_ID"])+1)):
                electrons_ID = self.track_data['Track_ID']==j # the data frame for each electron in each event
                each_df = self.track_data[event_select & electron_select & electrons_ID]
                Edep_df = self.track_data[event_select & electron_select & scintillator & electrons_ID]

                energy = Edep_df["DE"].sum()
                
                if sc_vol in each_df['Volume'].values and (each_df['Volume'].values)[-1] != sc_vol: #non-returning
                    non_returning += 1
                    non_returning_array.append(energy)
                    non_returning_event.append(i)
                    
                
                flag = 0  # A flag for determining whether it is straight e- or Returning e-
                
                if sc_vol in each_df['Volume'].values and (each_df['Volume'].values)[-1] == sc_vol: #contains returning and straight electron!
                    vol_array = each_df['Volume'].values
                    for k in range(len(vol_array)-1):
                        if vol_array[k] == sc_vol and vol_array[k+1]!=sc_vol:
                            flag=1
                    
                    if flag == 0:
                        straight_electron += 1
                        straight_electron_array.append(energy)
                        straight_event.append(i)

                    else:
                        returning_electron += 1
                        returning_electron_array.append(energy)
                        returning_electron_event.append(i)
                    
                if not sc_vol in each_df['Volume'].values:
                    never_electron +=1
                    never_array.append(energy)
                    never_event.append(i)
        
        straight_event = list(set(straight_event))
        non_returning_event=list(set(non_returning_event))
        returning_electron_event=list(set(returning_electron_event))
        never_event=list(set(never_event))
        print(f'time to count electrons {time.time() - t_start:.2f}')
        print('straight event', len(straight_event))
        print('non returning event', len(non_returning_event))
        print('returning electron event', len(returning_electron_event))
        print('never event', len(never_event))
        return np.array([straight_electron,returning_electron,non_returning,never_electron]),\
               [straight_electron_array,returning_electron_array,non_returning_array,never_array],\
               [straight_event,returning_electron_event,non_returning_event,never_event]
    
    
    def gaussian(self,x, a,mu,sig):
        return a/(math.sqrt(2.*math.pi)*sig)*np.exp(-np.power((x - mu)/sig, 2.)/2)

    def leastsquare(self,a,mu,sig):  #parameter: mu,sigma and (nu?! is nu = mu?!)
        return sum((self.gaussian(x,a,mu,sig)-y)**2 for x, y in zip(chosen_bin, chosen_yhist))
    
    def fit_peak(self,arg):
        
        yhist,binedges = np.histogram(self.area_collection,bins=100)
        bc = np.array((binedges[1:] + binedges[:-1])/2)
        yhist=np.array(yhist)/yhist.sum()
        
        global chosen_bin
        global chosen_yhist
        
        peak_index = np.argmax(yhist)
        left = 30
        right = len(yhist)-peak_index
        
        chosen_bin = bc[peak_index-left:peak_index+right]
        chosen_yhist= yhist[peak_index-left:peak_index+right]

        m = iminuit.Minuit(self.leastsquare,a=max(yhist),mu=bc[peak_index],sig=500,error_a=10, error_mu=100,error_sig=50,errordef=0.5)
        m.migrad() # run optimiser
        print(m.values)
        m.hesse()
        print(m.errors)  # {'x': 1,'y': 1,'z': 1}
        return m
    
    
def gaussian(x, a,mu,sig):
        return a/(math.sqrt(2.*math.pi)*sig)*np.exp(-np.power((x - mu)/sig, 2.)/2)

def leastsquare(a,mu,sig):  #parameter: mu,sigma and (nu?! is nu = mu?!)
    return sum((gaussian(x,a,mu,sig)-y)**2 for x, y in zip(chosen_bin, chosen_yhist))

def plot_spectra(obj,save_path):
    bin_number = 150
    
    area_straight_list = np.array([obj.area_collection[i] for i in obj.e_ID[0]])
    area_returning_list = [obj.area_collection[i] for i in obj.e_ID[1]] 
    area_non_list = [obj.area_collection[i] for i in obj.e_ID[2]]
    
    yhist,binedges = np.histogram(area_straight_list,bins=bin_number)
    yhistr,binedgesr = np.histogram(area_returning_list,bins=bin_number)
    yhistn,binedgesn = np.histogram(area_non_list,bins=bin_number)
    bc = np.array((binedges[1:] + binedges[:-1])/2)
    #plt.figure(figsize=(8,4.5))
    #plt.plot(binedges[1:],yhist,label='straight')
    #plt.plot(binedges[1:],yhistr,label='returning')
    #plt.plot(binedges[1:],yhistn,label='non-returning')
    #plt.legend()
    
    peak_index = np.argmax(yhist)
    left = 30
    right = len(yhist)-peak_index - 1

    global chosen_bin
    global chosen_yhist

    chosen_bin = bc[peak_index-left:peak_index+right]
    chosen_yhist= yhist[peak_index-left:peak_index+right]
    
    #m = iminuit.Minuit(leastsquare,a=max(yhist),mu=bc[peak_index],sig=500,error_a=100, error_mu=100,error_sig=500,errordef=0.5)
    #m = iminuit.Minuit(leastsquare,a=max(yhist),mu=bc[peak_index],sig=2000)
    #m.migrad()
    #m.hesse()
    #print(m.values)
    ##print(m.errors)

    x= np.linspace(5000,12000,60000)
    #y = gaussian(x,m.values[0],m.values[1],m.values[2])
    
    popt, pcov = curve_fit(gaussian, chosen_bin, chosen_yhist, p0=(max(yhist), bc[peak_index],2000))
    print('Fit results',*popt)
    perr = np.sqrt(np.diag(pcov))
    y = gaussian(x, *popt)
    plt.figure(figsize=(8,4.5))
    plt.scatter(bc,yhist,label = "Straight electrons distribusion")
    plt.scatter(chosen_bin,chosen_yhist,color='r',label="Fitting Data Points")
    plt.plot(x,y,label='Gaussian Fit',color = 'black')
    plt.axvline(bc[peak_index-left],color='red',linestyle=":")
    plt.axvline(bc[peak_index+right],color='red',linestyle=":",label = "Fitting Range")
    plt.ylim(0)
    plt.legend(prop={'size': 8})
    #plt.show()
    path = save_path+"plots/"
    if not os.path.exists(path):
        os.makedirs(path)

    plt.savefig(path+"SpectraFit_"+str(obj.pe)+"_pe_angle_"+str(obj.angle))
    
    stat = obj.e_stat
    print(" ** Summary of the electrons ** ")
    print("Portion of electron getting backscattered in scintillator",(stat[1]+stat[2])/np.sum(stat))
    print("Portion of Returning back-scattered electrons",(stat[1])/np.sum(stat))
    print("Portion of Non-returning back-scattered electrons",(stat[2])/np.sum(stat))
    print("Portion of undetected electrons",(stat[3])/np.sum(stat))

    ########################
    ########################
    
    fig, axs = plt.subplots(2,2,figsize=(13,8))
    axs[0,0].hist(area_straight_list,bins=bin_number,label = "Straight e-",histtype='step',color='g') #,normed=True
    axs[0,0].hist(area_returning_list,bins=bin_number,label = "Returning e-",histtype='step',color='b') #,normed=True
    axs[0,0].hist(area_non_list,bins=bin_number,label = "Non-retrurning",histtype='step',color='r') #,log=True   ,normed=True
    axs[0,0].plot(x,y,label="Gaussian Fit",color = 'black')
    axs[0,0].set_title("SiPM area count",fontsize=11)
    axs[0,0].legend(prop={'size': 8})
    axs[0,0].set(xlabel="Area")
    axs[0,0].relim()
    axs[0,0].autoscale_view()

    axs[1,0].hist(obj.Edep_dist[0],bins=bin_number,histtype='step',label = "Straight e-",color='g') #,log=True
    axs[1,0].hist(obj.Edep_dist[1],bins=bin_number,histtype='step',label = "Returning e-",color='b') #,log=True
    axs[1,0].hist(obj.Edep_dist[2],bins=bin_number,histtype='step',label = "Non-retrurning",color='r')
    axs[1,0].set_title("Energy depposited",fontsize=11)
    axs[1,0].set(xlabel="Energy (keV)")
    axs[1,0].legend(prop={'size': 8})

    axs[0,1].hist(area_straight_list,bins=bin_number,label = "Straight e-",histtype='step',color='g',log=True) #,normed=True
    axs[0,1].hist(area_returning_list,bins=bin_number,label = "Returning e-",histtype='step',color='b',log=True) #,normed=True
    axs[0,1].hist(area_non_list,bins=bin_number,label = "Non-retrurning",histtype='step',color='r',log=True) #,log=True   ,normed=True
    axs[0,1].set_title("SiPM area count (log-scale)",fontsize=11)
    axs[0,1].legend(prop={'size': 8})
    axs[0,1].set(xlabel='Area')

    axs[1,1].hist(obj.Edep_dist[0],bins=bin_number,histtype='step',color='g',label = "Straight e-",log=True) #,log=True
    axs[1,1].hist(obj.Edep_dist[1],bins=bin_number,histtype='step',color='b',label = "Returning e-",log=True) #,log=True
    axs[1,1].hist(obj.Edep_dist[2],bins=bin_number,histtype='step',color='r',label = "Non-retrurning",log=True) #,log=True
    axs[1,1].legend(prop={'size': 8})
    axs[1,1].set_title("Energy depposited (log-scale)",fontsize=11)
    axs[1,1].set(xlabel="Energy (keV)")

    fig.suptitle(str(obj.pe)+ " pe event Spectra at angle "+str(obj.angle)+" deg",fontsize=12)
    plt.subplots_adjust(hspace = 0.25)
        
    plt.savefig(path+"Spectra_"+str(obj.pe)+"_pe_angle_"+str(obj.angle))
        
    #plt.show()


if __name__=="__main__":
    main()
