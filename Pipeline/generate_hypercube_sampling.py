import numpy as np
import json
import copy
from smt.sampling_methods import LHS
import matplotlib.pyplot as plt

total_samples  = 550              # Total number of samples to generate
prefix         = "FOFR_linear_fixed_sigma8"         # A label for the simulations
dictfile       = "latin_hypercube_samples_"+prefix+"_"+str(total_samples)+".json" # Output dictionary file
make_hostfiles = True  # Make a list of hostfiles needed when running the sims
test_run       = False # Don't make dict, but plot samples instead...

# The nodes we want to use for running the sims on
# Important: Run the generate_input.py asking for the same number of nodes as in the list below
# E.g. if just euclid25 then run generate_input.py with nnodes=1
hostfiles = ["euclid25.uio.no","euclid26.uio.no","euclid27.uio.no","euclid28.uio.no","euclid29.uio.no","euclid30.uio.no"]

# Choose parameters to vary and the prior-range to vary
# We can look at e.g. the EuclidEmulator2 paper to pick the prior range for the emulator
parameters_to_vary = {
  'Omega_cdm':   [0.2,0.34],  # +-25%, euclid has 20%
  'w0':          [-1.3,-0.7], # +-30%, like Euclid
  'wa':          [-0.7,0.7],  # like Euclid
  'sigma8':      [0.66,0.98], # +-20%, more than Euclid
  'log10fofr0':  [-8.0,-4.0], 
}
parameters_to_vary_arr = []
for key in parameters_to_vary:
  parameters_to_vary_arr.append(key)

# Set the fiducial cosmology and simulations parameters
run_param_fiducial = {
  'label':        "FiducialCosmology",
  'outputfolder': "./output",
  'colaexe':      "../FML/FML/COLASolver/nbody",

  # COLA parameters
  'boxsize':    350.0,
  'Npart':      640,
  'Nmesh':      768,
  'Ntimesteps': 30,
  'Seed':       1234567,
  'zini':       30.0,
  'input_spectra_from_lcdm': "true",
  'sigma8_norm': "true",
  
  # Fiducial cosmological parameters - the ones we sample over will be changed below for each sample
  'cosmo_param': {
    # With physical parameters we specify Omega_i h^2 and h is a derived quantity
    # Otherwise we specify Omega_i and h and Omega_Lambda is a derived quantity
    'use_physical_parameters': False,
    'cosmology_model': 'w0waCDM',
    'gravity_model': 'f(R)',
    'h':          0.67,
    'Omega_b':    0.049,
    'Omega_cdm':  0.27,
    'Omega_ncdm': 0.001387,
    'Omega_k':    0.0,
    'omega_b':    0.049    * 0.67**2,
    'omega_cdm':  0.27     * 0.67**2,
    'omega_ncdm': 0.001387 * 0.67**2,
    'omega_k':    0.0      * 0.67**2,
    'omega_fld':  0.0      * 0.67**2,
    'w0':         -1.0, 
    'wa':         0.0,
    'Neff':       3.046,
    'k_pivot':    0.05,
    'A_s':        2.1e-9,
    'sigma8':     0.82,
    'n_s':        0.96,
    'T_cmb':      2.7255,
    'log10fofr0': -5.0,
    'screening': 'false',
    'screening_efficiency': 1.0,
    'largscale_linear': 'false',
    'kmax_hmpc':  20.0,
  },
 
  # Parameters used for generating a bash-file for running the sim
  'ncpu':     32, 
  'nthreads': 4,
  'npernode': 32,
}

# Generate nodelist
if make_hostfiles:
  count = 0
  for node in hostfiles:
    hostfile = run_param_fiducial['outputfolder'] + "/hostfile" + str(count) + ".txt"
    with open(hostfile,"w") as f:
      f.write(node)
    hostfiles[count] = hostfile
    count += 1

#========================================================================
#========================================================================

# Generate all samples
ranges = []
for key in parameters_to_vary:
  ranges.append(parameters_to_vary[key])
ranges = np.array(ranges)
sampling = LHS(xlimits=ranges)
all_samples = sampling(total_samples)

# Generate the dictionaries
sims = {}
count = 0
for sample in all_samples:
  if test_run:
    print("===========================")
    print("New parameter sample:")
  run_param = copy.deepcopy(run_param_fiducial)
  for i, param in enumerate(parameters_to_vary):
    run_param["cosmo_param"][param] = sample[i]
    if test_run:
      print("Setting ", param, " to value ", sample[i])
  label = prefix+str(count)
  run_param['label'] = label
  
  # For assigning different hostfiles to different nodes
  if make_hostfiles:
    node_number = count % len(hostfiles)
    run_param['hostfile'] = hostfiles[node_number]
    run_param['node_number'] = node_number

  sims[str(count)] = copy.deepcopy(run_param)
  count += 1

if test_run:
  nparam = len(parameters_to_vary_arr)
  for i in range(nparam):
    for j in range(i+1,nparam):
      plt.plot(all_samples[:,i], all_samples[:,j], "o")
      plt.xlabel(parameters_to_vary_arr[i])
      plt.ylabel(parameters_to_vary_arr[j])
      plt.show()
  exit(1)

# Save to file
with open(dictfile, "w") as f:
  data = json.dumps(sims)
  f.write(data)

