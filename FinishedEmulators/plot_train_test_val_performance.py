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
# Wrapper class we use to call the emulator
#=============================================
class PofkBoostEmulator:
    def __init__(self, path="", version=0):
        self.evaluator = EmulatorEvaluator.load(path + f"/version_{version}")
    def __call__(self, params):
        inputs = np.array(params)
        return self.evaluator(inputs).reshape(-1)

#=============================================
# Set the folder to the emulator and the version
#=============================================
emulator_folder = "./unscreened_fofr_lin_Pk/"
emulator_version = 0
pofkboostfunction = PofkBoostEmulator(path = emulator_folder, version = emulator_version)

#=============================================
# Redshifts for the uploaded emulators:
#=============================================
redshifts = ['30.000', '14.500', '9.333', '6.750', '5.200', 
             '4.167', '3.429', '2.875', '2.444', '2.100',
             '1.818', '1.583', '1.385', '1.214', '1.067',
             '0.938', '0.824', '0.722', '0.632', '0.550', 
             '0.476', '0.409', '0.348', '0.292', '0.240',
             '0.192', '0.148', '0.107', '0.069', '0.033', 
             '0.000']

#=============================================
# Load data. The "split" here is to fetch each block individually
#=============================================
data_       = np.genfromtxt('../Pipeline/neural_network_emulator/trainingdata/val.csv', delimiter=',')[1:]
data_train_ = np.genfromtxt('../Pipeline/neural_network_emulator/trainingdata/train.csv', delimiter=',')[1:]
data_test_  = np.genfromtxt('../Pipeline/neural_network_emulator/trainingdata/test.csv', delimiter=',')[1:]
split       = 384 # Length of k array basically

data        = []
data_train  = []
data_test   = []

#=============================================
# Desired redshift (for this test z = 0.0):
#=============================================
z_val = redshifts[-1]
for i in range(len(data_)//split):
    if (data_[i*split:(i+1)*split][0][5] == float(z_val)):
        data.append(data_[i*split:(i+1)*split])
for i in range(len(data_train_)//split):
    if (data_train_[i*split:(i+1)*split][0][5] == float(z_val)):
        data_train.append(data_train_[i*split:(i+1)*split])
for i in range(len(data_test_)//split):
    if (data_test_[i*split:(i+1)*split][0][5] == float(z_val)):
        data_test.append(data_test_[i*split:(i+1)*split])

#=============================================
# Make figures
#=============================================
colors   = ['#DDCC77','#882255','#6699CC']
size     = set_size(523.5307, dims=[8, 1], golden_ratio=True,fraction=0.5)
fig, axs = plt.subplots(8,1,figsize=(size[0],size[1]*0.53),\
                        gridspec_kw={'height_ratios':[1.0,0.4,0.05,1.0,0.4,0.05,1.0,0.4]},sharex=True)

for i, data_i in enumerate(data_train): 

    #=============================================
    # Extract data from train file
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

for i, data_i in enumerate(data): 
    
    #=============================================
    # Extract data from val file
    #=============================================
    log10k = data_i[:,-2]
    k      = 10**log10k
    boost  = data_i[:,-1]
    axs[3].semilogx(k,boost,color=colors[2],alpha=0.5,)

    #=============================================
    # Fetch parameters from the validation set
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
    axs[3].semilogx(k, boost_emulator, color=colors[1], linewidth=0.7, alpha=0.5)
    axs[4].semilogx(k, boost_emulator/boost - 1.0, color=colors[0], linewidth=0.7, alpha=0.5)

for i, data_i in enumerate(data_test): 
    
    #=============================================
    # Extract data from test file
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
    
    # Make plot
    axs[6].semilogx(k, boost_emulator, color=colors[1], linewidth=0.7, alpha=0.5)
    axs[7].semilogx(k, boost_emulator/boost - 1.0, color=colors[0], linewidth=0.7, alpha=0.5)

#=============================================
# Prettify the plots
#=============================================
axs[1].set_ylabel(r'$\displaystyle \mathrm{Difference}$', fontsize=7)
axs[0].set_ylabel(r"$r(k,z) = P_{f(R)}/P_{\rm GR}$", fontsize=7)
axs[4].set_ylabel(r'$\displaystyle \mathrm{Difference}$', fontsize=7)
axs[3].set_ylabel(r"$r(k,z) = P_{f(R)}/P_{\rm GR}$", fontsize=7)
axs[7].set_ylabel(r'$\displaystyle \mathrm{Difference}$', fontsize=7)
axs[6].set_ylabel(r"$r(k,z) = P_{f(R)}/P_{\rm GR}$", fontsize=7)
axs[7].set_xlabel(r'$k\,\;[h^{-1}\mathrm{Mpc}]$')

axs[1].set_ylim(-0.03,0.03)
axs[4].set_ylim(-0.03,0.03)
axs[7].set_ylim(-0.03,0.03)
axs[0].set_ylim(0.97,1.63)
axs[3].set_ylim(0.97,1.63)
axs[6].set_ylim(0.97,1.63)

axs[2].set_visible(False)
axs[5].set_visible(False)

axs[1].fill_between(k, k*0-0.01,k*0+0.01,color='gray',alpha=0.15,linewidth=0.0)
axs[4].fill_between(k, k*0-0.01,k*0+0.01,color='gray',alpha=0.15,linewidth=0.0)
axs[7].fill_between(k, k*0-0.01,k*0+0.01,color='gray',alpha=0.15,linewidth=0.0)

axs[0].set_xlim(min(k),max(k))

data_patch = mlines.Line2D([0,0],[0,0],color=colors[2], alpha=0.5,label='data')
emu_patch  = mlines.Line2D([0,0],[0,0],color=colors[1],alpha=0.5, label='emulator')
axs[0].legend(handles=[data_patch,emu_patch],frameon=False,fontsize=7,labelspacing=0.2,loc='upper left')

axs[0].text(0.03,1.4,'Training set')
axs[0].text(0.03,1.36,'z = '+z_val)
axs[3].text(0.03,1.4,'Validation set')
axs[3].text(0.03,1.36,'z = '+z_val)
axs[6].text(0.03,1.4,'Test set')
axs[6].text(0.03,1.36,'z = '+z_val)

plt.subplots_adjust(wspace=0, hspace=0)
fig.align_ylabels()

#=============================================
# Show plot
#=============================================
plt.show()

#=============================================
# Save plot as a PDF
#=============================================
figpath = f'cm_ver{emulator_version}_linear_z_'+z_val+'_train_val_test.pdf'
print("Saving plot to ", figpath)
fig.savefig(figpath, format='pdf', bbox_inches='tight')
