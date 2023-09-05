import numpy as np
import json
import glob
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import random
from scipy.signal import savgol_filter
import random

def get_available_redshifts():
    folder_path    = '/mn/stornext/d13/euclid/renatmau/PhD/own_project_runs/paper2/COLA_Pk_ratio/linear_sims_run_separated/testing/output/'
    file_pattern   = 'pofk_FOFR_linear_fixed_sigma8_test0_cb_z*.txt'
    matching_files = glob.glob(folder_path + '/' + file_pattern)
    numbers_str    = []
    numbers        = []
    for file in matching_files:
        number = file.split('_z')[1].split('.txt')[0]
        numbers.append(float(number))
        numbers_str.append(str(number))
    # sort from highest to lowest
    numbers, numbers_str = zip(*sorted(zip(numbers, numbers_str), reverse=True))
    return numbers_str

def spline_ratio(k,ratio):
    cs        = CubicSpline(k, ratio)
    new_k     = np.logspace(np.log10(min(k)),np.log10(max(k)),len(k))
    new_ratio = cs(new_k)
    return new_k, new_ratio

def read_file(N_sample,z,set_name,set_prefix):
    home      = '/mn/stornext/d13/euclid/renatmau/PhD/own_project_runs/paper2/COLA_Pk_ratio/linear_sims_run_separated/'+set_name+'/output/'
    ratio     = []
    ratio_lin = []
    #read data:
    k,Pk,Pk_lin          = np.loadtxt(home+'pofk_FOFR_linear_fixed_sigma8_'+set_prefix+str(N_sample)+'_cb_z'+z+'.txt',unpack=True)
    k_GR,Pk_GR,Pk_lin_GR = np.loadtxt(home+'pofk_FOFR_linear_fixed_sigma8_'+set_prefix+str(N_sample)+'_GR_cb_z'+z+'.txt',unpack=True)
    #calculate ratios:
    r     = Pk/Pk_GR
    r_lin = Pk_lin/Pk_lin_GR
    #spline to get log-spaced k-array:
    new_k, new_r         = spline_ratio(k,r)
    new_k_lin, new_r_lin = spline_ratio(k,r_lin)
    #smooth data:
    new_r     = savgol_filter(new_r, 25, 3)
    new_r_lin = savgol_filter(new_r_lin, 25, 3)
    #append all samples to array:
    ratio.append(new_k)
    ratio.append(new_r)
    ratio_lin.append(new_k_lin)
    ratio_lin.append(new_r_lin)
    return ratio_lin

def read_json(filename):
    # Read the JSON data from a file
    with open(filename) as file:
        json_data = json.load(file)

    s8 = []; Omegacdm = []; w0 = []; wa = []; log10fofr = []

    for entry in json_data.values():
        s8.append(entry['cosmo_param']['sigma8'])
        Omegacdm.append(entry['cosmo_param']['Omega_cdm'])
        w0.append(entry['cosmo_param']['w0'])
        wa.append(entry['cosmo_param']['wa'])
        log10fofr.append(entry['cosmo_param']['log10fofr0'])   
    
    return s8, Omegacdm, w0, wa, log10fofr


redshifts   = get_available_redshifts()
sets_name   = ['training','testing','validation']
sets_prefix = ['train','test','val']
sets_num    = [440,55,55] 

for i in range(len(sets_name)):

    s8_list = []; Omegacdm_list  = []; w0_list = []
    wa_list = []; log10fofr_list = []; z_list  = []
    k_list  = []; ratio_list     = []

    jsonfile  = '/mn/stornext/d13/euclid/renatmau/PhD/own_project_runs/paper2/COLA_Pk_ratio/linear_sims_run_separated/'+sets_name[i]+'/latin_hypercube_samples_FOFR_linear_fixed_sigma8_'+sets_prefix[i]+'_'+str(int(sets_num[i]))+'.json'

    s8, Omegacdm, w0, wa, log10fofr = read_json(jsonfile)
    headers = ["sigma8", "Omegacdm", "w0", "wa", "log10fofr", "z", "log10k", "boost"]

    for z in range(len(redshifts)):
        for s in range(sets_num[i]):

            print ('Appending redshift ', redshifts[z], ' for sample ', s, 'for data set ', sets_name[i])
            
            ratio = read_file(s,redshifts[z],sets_name[i],sets_prefix[i])

            s8_list.extend(np.zeros(len(ratio[0]))+s8[s])
            Omegacdm_list.extend(np.zeros(len(ratio[0]))+Omegacdm[s])
            w0_list.extend(np.zeros(len(ratio[0]))+w0[s])
            wa_list.extend(np.zeros(len(ratio[0]))+wa[s])
            log10fofr_list.extend(np.zeros(len(ratio[0]))+log10fofr[s])
            z_list.extend(np.zeros(len(ratio[0]))+float(redshifts[z]))
            k_list.extend(np.log10(ratio[0]))
            ratio_list.extend(ratio[1])

    data = np.zeros((len(s8_list),len(headers)))
    data[:,0] = s8_list
    data[:,1] = Omegacdm_list
    data[:,2] = w0_list
    data[:,3] = wa_list
    data[:,4] = log10fofr_list
    data[:,5] = z_list
    data[:,6] = k_list
    data[:,7] = ratio_list

    np.savetxt('lin_separatedLHS_'+sets_prefix[i]+'.csv', data, delimiter=",", header=",".join(headers), comments="")

