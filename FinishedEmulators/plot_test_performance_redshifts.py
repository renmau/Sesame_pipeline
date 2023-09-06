import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from figure_size import set_size
import matplotlib.lines as mlines
from Emulator import EmulatorEvaluator

#=============================================
# Set plotting defaults
#=============================================
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams.update({'font.size': 7})
matplotlib.rcParams['text.usetex'] = True
params = {'xtick.top': True, 'ytick.right': True, 'xtick.direction': 'in', 'ytick.direction': 'in'}
plt.rcParams.update(params)

#=============================================
# A simple class for extracting the boost 
#=============================================
class PofkBoostEmulator:
    def __init__(self, path="", version=0):
        self.evaluator = EmulatorEvaluator.load(path + f"/version_{version}")
    def __call__(self, params):
        inputs = np.array(params)
        return self.evaluator(inputs).reshape(-1)

#=============================================
# Set the folder to the emulator and the version
# and set up the boost-function
#=============================================
emulator_folder = "unscreened_fofr_lin_Pk/"
emulator_version = 0
pofkboostfunction = PofkBoostEmulator(path = emulator_folder, version = emulator_version)

#=============================================
# Could get redshift from data, but since we dont 
# have all the data uploaded here, these are the available
# redshifts for the uploaded emulators:
#=============================================
redshifts = ['30.000', '14.500', '9.333', '6.750', '5.200', 
             '4.167', '3.429', '2.875', '2.444', '2.100', 
             '1.818', '1.583', '1.385', '1.214', '1.067',
             '0.938', '0.824', '0.722', '0.632', '0.550', 
             '0.476', '0.409', '0.348', '0.292', '0.240', 
             '0.192', '0.148', '0.107', '0.069', '0.033', 
             '0.000']
z_val = [redshifts[-20],redshifts[-12],redshifts[-1]]

#=============================================
# Load data. The "split" here is to fetch each 
# "P(k)/P(k)_GR" block individually
#=============================================
data_test_ = np.genfromtxt('../Pipeline/neural_network_emulator/trainingdata/test.csv', delimiter=',')[1:]
split      = 384 # Length of k array
data_test0 = []; data_test1  = [];data_test2  = []
for i in range(len(data_test_)//split):
    if (data_test_[i*split:(i+1)*split][0][5] == float(z_val[0])):
        data_test0.append(data_test_[i*split:(i+1)*split])
    if (data_test_[i*split:(i+1)*split][0][5] == float(z_val[1])):
        data_test1.append(data_test_[i*split:(i+1)*split])
    if (data_test_[i*split:(i+1)*split][0][5] == float(z_val[2])):
        data_test2.append(data_test_[i*split:(i+1)*split])

#=============================================
# Set up figures
#=============================================
colors   = ['#DDCC77','#882255','#6699CC']
size     = set_size(523.5307, dims=[8, 1], golden_ratio=True,fraction=0.5)
fig, axs = plt.subplots(8,1,figsize=(size[0],size[1]*0.53),\
                        gridspec_kw={'height_ratios':[1.0,0.4,0.05,1.0,0.4,0.05,1.0,0.4]},sharex=True)

for i, data_i in enumerate(data_test0): 
    
    #=============================================
    # Extract data from test file, first redshift
    #=============================================
    log10k = data_i[:,-2]
    k      = 10**log10k
    boost  = data_i[:,-1]
    axs[0].semilogx(k,boost,color=colors[2],alpha=0.5,)

    #=============================================
    # Fetch parameters from the trainingset
    #=============================================
    params_cosmo = np.array([
        data_i[0,0],
        data_i[0,1],    
        data_i[0,2],
        data_i[0,3],
        data_i[0,4],
        data_i[0,5],]) 
    params_batch = np.column_stack(( np.vstack([params_cosmo] * len(log10k)), log10k))
    
    #=============================================
    # Call emulator
    #=============================================
    boost_emulator = pofkboostfunction(params_batch)
    
    #=============================================
    # Make plot
    #=============================================
    axs[0].semilogx(k, boost_emulator, color=colors[1], linewidth=0.7, alpha=0.5)
    axs[1].semilogx(k, boost_emulator/boost - 1.0, color=colors[0], linewidth=0.7, alpha=0.5)

for i, data_i in enumerate(data_test1): 
    
    #================================================
    # Extract data from test file, second redshift
    #================================================
    log10k = data_i[:,-2]
    k      = 10**log10k
    boost  = data_i[:,-1]
    axs[3].semilogx(k,boost,color=colors[2],alpha=0.5,)
    
    #================================================
    # Fetch parameters from the validation set
    #================================================
    params_cosmo = np.array([
          data_i[0,0],
          data_i[0,1],    
          data_i[0,2],
          data_i[0,3],
          data_i[0,4],
          data_i[0,5],])
    params_batch = np.column_stack(( np.vstack([params_cosmo] * len(log10k)), log10k))
    
    #================================================
    # Call emulator
    #================================================
    boost_emulator = pofkboostfunction(params_batch)
    
    #=============================================
    # Make plot
    #=============================================
    axs[3].semilogx(k, boost_emulator, color=colors[1], linewidth=0.7, alpha=0.5)
    axs[4].semilogx(k, boost_emulator/boost - 1.0, color=colors[0], linewidth=0.7, alpha=0.5)

for i, data_i in enumerate(data_test2): 
    
    #=============================================
    # Extract data from test file, third redshift
    #=============================================
    log10k = data_i[:,-2]
    k      = 10**log10k
    boost  = data_i[:,-1]
    axs[6].semilogx(k,boost,color=colors[2],alpha=0.5,)
    
    #=============================================
    # Fetch parameters from the test set
    #=============================================
    params_cosmo = np.array([
          data_i[0,0],
          data_i[0,1],    
          data_i[0,2],
          data_i[0,3],
          data_i[0,4],
          data_i[0,5],])
    params_batch = np.column_stack(( np.vstack([params_cosmo] * len(log10k)), log10k))
    
    #=============================================
    # Call emulator
    #=============================================
    boost_emulator = pofkboostfunction(params_batch)
    
    #=============================================
    # Make plot
    #=============================================
    axs[6].semilogx(k, boost_emulator, color=colors[1], linewidth=0.7, alpha=0.5)
    axs[7].semilogx(k, boost_emulator/boost - 1.0, color=colors[0], linewidth=0.7, alpha=0.5)

#=============================================
# Prettify the plots
#=============================================
axs[1].set_ylabel(r'$\displaystyle \mathrm{Rel. Diff.}$', fontsize=7)
axs[0].set_ylabel(r"$B(k,z) = P_{f(R)}/P_{\rm GR}$", fontsize=7)
axs[4].set_ylabel(r'$\displaystyle \mathrm{Rel. Diff.}$', fontsize=7)
axs[3].set_ylabel(r"$B(k,z) = P_{f(R)}/P_{\rm GR}$", fontsize=7)
axs[7].set_ylabel(r'$\displaystyle \mathrm{Rel. Diff.}$', fontsize=7)
axs[6].set_ylabel(r"$B(k,z) = P_{f(R)}/P_{\rm GR}$", fontsize=7)

axs[1].set_ylim(-0.045,0.045)
axs[4].set_ylim(-0.045,0.045)
axs[7].set_ylim(-0.045,0.045)
axs[0].set_ylim(0.97,1.63)
axs[3].set_ylim(0.97,1.63)
axs[6].set_ylim(0.97,1.63)

axs[7].set_xlabel(r'$k\,\;[h\,\mathrm{Mpc}^{-1}]$')

axs[2].set_visible(False)
axs[5].set_visible(False)

axs[1].fill_between(k, k*0-0.01,k*0+0.01,color='gray',alpha=0.15,linewidth=0.0)
axs[4].fill_between(k, k*0-0.01,k*0+0.01,color='gray',alpha=0.15,linewidth=0.0)
axs[7].fill_between(k, k*0-0.01,k*0+0.01,color='gray',alpha=0.15,linewidth=0.0)

axs[0].set_xlim(min(k),max(k))

data_patch = mlines.Line2D([0,0],[0,0],color=colors[2], alpha=0.5,label='data')
emu_patch  = mlines.Line2D([0,0],[0,0],color=colors[1],alpha=0.5, label='emulator')
axs[0].legend(handles=[data_patch,emu_patch],frameon=False,fontsize=7,labelspacing=0.2,loc='upper left')

axs[0].text(0.03,1.36,'z = '+z_val[0])
axs[3].text(0.03,1.36,'z = '+z_val[1])
axs[6].text(0.03,1.36,'z = '+z_val[2])

plt.subplots_adjust(wspace=0, hspace=0)
fig.align_ylabels()

#=============================================
# Show plot
#=============================================
plt.show()

#=============================================
# Save the plot as PDFs
#=============================================
figpath = f'cm_ver{emulator_version}_lin_fofr_linPk_3zs.pdf'
print("Saving plot to ", figpath)
fig.savefig(figpath, format='pdf', bbox_inches='tight')
