import numpy as np
import json
import glob
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import random
from scipy.signal import savgol_filter
import random

def get_available_redshifts():
    folder_path    = '/mn/stornext/d13/euclid/renatmau/PhD/own_project_runs/paper2/COLA_Pk_ratio/linear_sims_run/output/'
    file_pattern   = 'pofk_FOFR_linear_fixed_sigma80_cb_z*.txt'
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
    cs = CubicSpline(k, ratio)
    new_k = np.logspace(np.log10(min(k)),np.log10(max(k)),len(k))
    new_ratio = cs(new_k)
    return new_k, new_ratio

def read_file(N_sample,z):
    home      = '/mn/stornext/d13/euclid/renatmau/PhD/own_project_runs/paper2/COLA_Pk_ratio/linear_sims_run/output/'
    ratio     = []
    ratio_lin = []
    #read data:
    k,Pk,Pk_lin          = np.loadtxt(home+'pofk_FOFR_linear_fixed_sigma8'+str(N_sample)+'_cb_z'+z+'.txt',unpack=True)
    k_GR,Pk_GR,Pk_lin_GR = np.loadtxt(home+'pofk_FOFR_linear_fixed_sigma8'+str(N_sample)+'_GR_cb_z'+z+'.txt',unpack=True)
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
    return ratio, ratio_lin

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


redshifts = get_available_redshifts()
#redshifts = ['0.000']
jsonfile  = '/mn/stornext/d13/euclid/renatmau/PhD/own_project_runs/paper2/COLA_Pk_ratio/linear_sims_run/latin_hypercube_samples_FOFR_linear_fixed_sigma8_550.json'

#make list with available samples:
tot_samples       = 550
samples_avail     = list(range(tot_samples))
num_random_combos = tot_samples*len(redshifts)
random_numbers    = [random.random() for _ in range(num_random_combos)]


s8_train = []; s8_val = []; s8_test = []
Omegacdm_train = []; Omegacdm_val = []; Omegacdm_test = []
w0_train = []; w0_val = []; w0_test = []
wa_train = []; wa_val = []; wa_test = []
log10fofr_train = []; log10fofr_val = []; log10fofr_test = []
z_train = []; z_val = []; z_test = []
k_train = []; k_val = []; k_test = []
ratio_train = []; ratio_val = []; ratio_test = []

s8, Omegacdm, w0, wa, log10fofr = read_json(jsonfile)
headers = ["sigma8", "Omegacdm", "w0", "wa", "log10fofr", "z", "log10k", "boost"]

for i in range(tot_samples):
    for z in range(len(redshifts)):
            ratio, ratio_lin = read_file(i,redshifts[z])

            random_num = random.choice(random_numbers) #pick a random number from the array
            random_numbers.remove(random_num) # remove that number
            
            if (random_num < 0.8):
                print ('sample ', i, ' redshift ', redshifts[z],' goes into training file')
                s8_train.extend(np.zeros(len(ratio[0]))+s8[i])
                Omegacdm_train.extend(np.zeros(len(ratio[0]))+Omegacdm[i])
                w0_train.extend(np.zeros(len(ratio[0]))+w0[i])
                wa_train.extend(np.zeros(len(ratio[0]))+wa[i])
                log10fofr_train.extend(np.zeros(len(ratio[0]))+log10fofr[i])
                z_train.extend(np.zeros(len(ratio[0]))+float(redshifts[z]))
                k_train.extend(np.log10(ratio[0]))
                ratio_train.extend(ratio[1])

            if (random_num >= 0.8) and (random_num < 0.9):
                print ('sample ', i, ' redshift ', redshifts[z],' goes into validation file')
                s8_val.extend(np.zeros(len(ratio[0]))+s8[i])
                Omegacdm_val.extend(np.zeros(len(ratio[0]))+Omegacdm[i])
                w0_val.extend(np.zeros(len(ratio[0]))+w0[i])
                wa_val.extend(np.zeros(len(ratio[0]))+wa[i])
                log10fofr_val.extend(np.zeros(len(ratio[0]))+log10fofr[i])
                z_val.extend(np.zeros(len(ratio[0]))+float(redshifts[z]))
                k_val.extend(np.log10(ratio[0]))
                ratio_val.extend(ratio[1])

            if (random_num >= 0.9):
                print ('sample ', i, ' redshift ', redshifts[z],' goes into testing file')
                s8_test.extend(np.zeros(len(ratio[0]))+s8[i])
                Omegacdm_test.extend(np.zeros(len(ratio[0]))+Omegacdm[i])
                w0_test.extend(np.zeros(len(ratio[0]))+w0[i])
                wa_test.extend(np.zeros(len(ratio[0]))+wa[i])
                log10fofr_test.extend(np.zeros(len(ratio[0]))+log10fofr[i])
                z_test.extend(np.zeros(len(ratio[0]))+float(redshifts[z]))
                k_test.extend(np.log10(ratio[0]))
                ratio_test.extend(ratio[1])
            

data_train = np.zeros((len(s8_train),len(headers)))
data_train[:,0] = s8_train
data_train[:,1] = Omegacdm_train
data_train[:,2] = w0_train
data_train[:,3] = wa_train
data_train[:,4] = log10fofr_train
data_train[:,5] = z_train
data_train[:,6] = k_train
data_train[:,7] = ratio_train

data_val = np.zeros((len(s8_val),len(headers)))
data_val[:,0] = s8_val
data_val[:,1] = Omegacdm_val
data_val[:,2] = w0_val
data_val[:,3] = wa_val
data_val[:,4] = log10fofr_val
data_val[:,5] = z_val
data_val[:,6] = k_val
data_val[:,7] = ratio_val

data_test = np.zeros((len(s8_test),len(headers)))
data_test[:,0] = s8_test
data_test[:,1] = Omegacdm_test
data_test[:,2] = w0_test
data_test[:,3] = wa_test
data_test[:,4] = log10fofr_test
data_test[:,5] = z_test
data_test[:,6] = k_test
data_test[:,7] = ratio_test

print ('Percentage of samples going into train:')
print (float(len(s8_train))/float((len(s8_train)+len(s8_val)+len(s8_test))))
print ('Percentage of samples going into val:')
print (float(len(s8_val))/float((len(s8_train)+len(s8_val)+len(s8_test))))
print ('Percentage of samples going into test:')
print (float(len(s8_test))/float((len(s8_train)+len(s8_val)+len(s8_test))))
'''
Prints percentages going into train, val and test:
0.8029325513196481
0.0978299120234604
0.0992375366568915
'''

np.savetxt('randomization_train.csv', data_train, delimiter=",", header=",".join(headers), comments="")
np.savetxt('randomization_val.csv', data_val, delimiter=",", header=",".join(headers), comments="")
np.savetxt('randomization_test.csv', data_test, delimiter=",", header=",".join(headers), comments="")


