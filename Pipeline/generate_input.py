from classy import Class
import subprocess
import numpy as np
import json
import sys
import os

# Make the bashscript or not? In not then one must run the COLA sims yourself (paramfiles will be made and placed in output folder)
generate_bash_files = True

# Get name of dictionary file, the prefix for the bashscript we make and the number of nodes we have to run on
# We are currently assuming one sim can be run on one node here
if len(sys.argv) < 4:
  print("Error: missing arguments")
  exit(1)
dictfile       = sys.argv[1]
bashfileprefix = sys.argv[2]
nnodes         = int(sys.argv[3])

# Make list of bashfiles corresponding to the number of nodes we have availiable
if generate_bash_files:
  bashfiles = []
  for i in range(nnodes):
    curbashfile = bashfileprefix + str(i) + ".sh"
    bashfiles.append(curbashfile)
    subprocess.run(["touch", curbashfile])
else:
  bashfiles = None

# Load dict-of-dict where dict["i"] gives parameters for sim "i"
data = []
with open(dictfile) as json_file:
  data = json.load(json_file)

class Generate:

  def __init__(self, run_param, bashfiles = None, debug = False):
    self.run_param = run_param
    self.bashfiles = bashfiles
    self.debug = debug

    # Label for the run
    self.label = run_param['label']
    self.label_GR = self.label + "_GR"
    
    # Folder to store everything (use the full path here)
    self.outputfolder = run_param['outputfolder']
    if not os.path.isdir(self.outputfolder):
      subprocess.run(["mkdir", self.outputfolder])
    self.colaexe = run_param['colaexe']
    
    # CLASS parameters (passed to CLASS in [3] below so check that is correct)
    cosmo_param = run_param["cosmo_param"]
    self.h          = cosmo_param['h']           # Hubble H0 / 100km/s/Mpc
    self.use_physical_parameters = cosmo_param['use_physical_parameters']
    if self.use_physical_parameters:
      self.omega_b    = cosmo_param['omega_b']     # Baryon density
      self.omega_cdm  = cosmo_param['omega_cdm']   # CDM density
      self.omega_ncdm = cosmo_param['omega_ncdm']  # Massive neutrino density
      self.omega_k    = cosmo_param['omega_k']     # Curvature density
      self.omega_fld  = cosmo_param['omega_fld']   # Dark energy density
      self.Omega_b    = self.omega_b / self.h**2
      self.Omega_cdm  = self.omega_cdm / self.h**2
      self.Omega_ncdm = self.omega_ncdm / self.h**2
      self.Omega_k    = self.omega_k / self.h**2
      self.Omega_fld  = 1.0 - (self.Omega_b+self.Omega_cdm+self.Omega_ncdm+self.Omega_k)
    else:
      self.Omega_b    = cosmo_param['Omega_b']     # Baryon density
      self.Omega_cdm  = cosmo_param['Omega_cdm']   # CDM density
      self.Omega_ncdm = cosmo_param['Omega_ncdm']  # Massive neutrino density
      self.Omega_k    = cosmo_param['Omega_k']     # Curvature density
      self.Omega_fld  = 1.0 - (self.Omega_b+self.Omega_cdm+self.Omega_ncdm+self.Omega_k)
      self.omega_fld  = self.Omega_fld * self.h**2
      self.omega_b    = self.Omega_b * self.h**2
      self.omega_cdm  = self.Omega_cdm * self.h**2
      self.omega_ncdm = self.Omega_ncdm * self.h**2
      self.omega_k    = self.Omega_k * self.h**2

    self.w0_fld     = cosmo_param['w0']          # CPL parametrization
    self.wa_fld     = cosmo_param['wa']          # CPL parametrization
    self.A_s        = cosmo_param['A_s']         # Spectral ampltitude
    self.sigma8     = cosmo_param['sigma8']      
    self.n_s        = cosmo_param['n_s']         # Spectral index
    self.k_pivot    = cosmo_param['k_pivot']     # Pivot scale in 1/Mpc
    self.T_cmb      = cosmo_param['T_cmb']       # CMB temperature
    self.Neff       = cosmo_param['Neff']
    self.N_ncdm     = 1
    self.N_ur       = self.Neff-self.N_ncdm                # Effective number of MASSLESS neutrino species
    try:
      self.kmax_hmpc = cosmo_param['kmax_hmpc']
    except:
      self.kmax_hmpc = 20.0

    # Fixed COLA parameters
    self.boxsize    = run_param['boxsize']
    self.npart      = run_param['Npart']
    self.nmesh      = run_param['Nmesh']
    self.ntimesteps = run_param['Ntimesteps']
    self.seed       = run_param['Seed']
    self.zini       = run_param['zini']
    self.kcola      = "true" if self.Omega_ncdm > 0. else "false"
    self.cosmology  = cosmo_param['cosmology_model']
    self.gravity    = cosmo_param['gravity_model']
    
    # Some other options for cosmologies and gravities
    if self.gravity == "f(R)":
      self.fofr0 = 10**cosmo_param['log10fofr0']
      self.screening_efficiency = cosmo_param['screening_efficiency']
      self.largscale_linear = cosmo_param['largscale_linear']
      self.screening = cosmo_param['screening']
    else:
      self.fofr0 = 0.0

    self.nthreads   = run_param['nthreads']
    self.ncpu       = run_param['ncpu']
    self.npernode   = run_param['npernode']
    self.hostfile   = run_param['hostfile']
    self.node_number= run_param['node_number']
    
    # If we only have CLASS input for GR then we can use this
    # and the code will compute growth-factors and scale it accordingly
    # If MG parameters is passed to hi-class and computed exactly then 
    # you have to use "false" below.
    self.input_spectra_from_lcdm = run_param['input_spectra_from_lcdm']
    # if we want to normalize sigma_8 (at redshift 0.0 at sigma8=0.83) or not:
    self.sigma8_norm = run_param['sigma8_norm']

  def generate_data(self, GR):
    self.name = self.label
    if GR:
      self.name = self.label_GR

    # In case of GR use gravity model GR
    gravity = self.gravity
    cosmology = self.cosmology
    if GR:
      cosmology = "w0waCDM"
      gravity   = "GR"

    print("Doing model ", self.name, " Gravity: ", gravity, " Cosmology: ", cosmology)

    # [1] Make list of redshifts 
    zarr = np.exp(-np.linspace(np.log(1.0/(1.0+self.zini)),np.log(1.0),100))-1.0
    zarr = np.flip(zarr)
    for i in range(len(zarr)):
      zarr[i] = round(zarr[i],3)
    zlist = str(zarr[0])
    for i in range(1,len(zarr)):
      zlist += ","+str(zarr[i])
    
    # [2] Set class parameters and run class
    params = {
        'root': self.outputfolder+'/class_'+self.name+'_',
        #'write_parameters': 'yes',
        'format': 'camb',
        'output': 'tCl,mPk,mTk',
        'l_max_scalars': 2000,
        'P_k_max_1/Mpc': self.kmax_hmpc,
        'h': self.h,
        'w0_fld': self.w0_fld,
        'wa_fld': self.wa_fld,
        'A_s': self.A_s,
        'n_s': self.n_s, 
        'k_pivot': self.k_pivot,
        'N_ur': self.N_ur,
        'N_ncdm': self.N_ncdm,
        'T_cmb': self.T_cmb,
        'z_pk': zlist,
    }
    if self.use_physical_parameters:
      params['omega_b'] = self.omega_b
      params['omega_cdm'] = self.omega_cdm
      params['omega_k'] = self.omega_k
      params['omega_ncdm'] = self.omega_ncdm
      params['omega_fld'] = self.omega_fld
    else:
      params['Omega_b'] = self.Omega_b
      params['Omega_cdm'] = self.Omega_cdm
      params['Omega_k'] = self.Omega_k
      params['Omega_ncdm'] = self.Omega_ncdm
      params['Omega_fld'] = self.Omega_fld

    if self.debug:
      print(params)
    cosmo = Class()
    cosmo.set(params)
    cosmo.compute()
    
    # [3] Extract and output transfer information (in CAMB format suitable for COLA)
    colatransferinfofile = self.outputfolder + " " + str(len(zarr))
    for _z in zarr:
      # Transfer
      transfer = cosmo.get_transfer(z=_z, output_format="camb")
      cols = ['k (h/Mpc)', '-T_cdm/k2', '-T_b/k2', '-T_g/k2', '-T_ur/k2', '-T_ncdm/k2', '-T_tot/k2']
      
      # Pofk
      karr = transfer['k (h/Mpc)']
      pofk_total = np.array([cosmo.pk(_k * self.h, _z)*self.h**3 for _k in karr])
      pofk_cb = np.array([cosmo.pk_cb(_k * self.h, _z)*self.h**3 for _k in karr])
      pofkfilename = "class_power_"+self.name+"_z"+str(_z)+".txt"
      np.savetxt(self.outputfolder + "/" + pofkfilename, np.c_[karr, pofk_cb, pofk_total], header="# k (h/Mpc)  P_tot(k)   P_cb(k)  (Mpc/h)^3")
  
      nextra = 13 - len(cols) # We need 13 columns in input so just add some extra zeros
      output = []
      for col in cols:
        output.append(transfer[col])
      for i in range(nextra):
        output.append(transfer[cols[0]]*0.0)
      output = np.array(output)
      filename = "class_transfer_"+self.name+"_z"+str(_z)+".txt"
      np.savetxt(self.outputfolder + "/" + filename, np.c_[output.T], 
                 header="0: k/h   1: CDM      2: baryon   3: photon  4: nu     5: mass_nu  6: total " +
                 "*7: no_nu *8: total_de *9: Weyl *10: v_CDM *11: v_b *12: (v_b-v_c) (* Not present) simlabel = [" + self.name + "]")
      colatransferinfofile += "\n" + filename + " " + str(_z) 
    
    # [4] Write transferinfo-file needed by COLA
    colatransferinfofilename = self.outputfolder + "/class_transferinfo_"+self.name+".txt"
    with open(colatransferinfofilename, 'w') as f:
        f.write(colatransferinfofile)
    
    # [6] Write the COLA parameterfile
    colafile = "\
------------------------------------------------------------ \n\
-- Simulation parameter file                                 \n\
-- Include other paramfile into this: dofile(\"param.lua\")  \n\
------------------------------------------------------------ \n\
                                                             \n\
-- Don't allow any parameters to take optional values?       \n\
all_parameters_must_be_in_file = true                        \n\
------------------------------------------------------------ \n\
-- Simulation options                                        \n\
------------------------------------------------------------ \n\
-- Label                                                     \n\
simulation_name = \""+self.name+"\"                          \n\
-- Boxsize of simulation in Mpc/h                            \n\
simulation_boxsize = "+str(self.boxsize)+"                   \n\
                                                             \n\
------------------------------------------------------------ \n\
-- COLA                                                      \n\
------------------------------------------------------------ \n\
-- Use the COLA method                                       \n\
simulation_use_cola = true                                   \n\
simulation_use_scaledependent_cola = "+self.kcola+"          \n\
                                                             \n\
------------------------------------------------------------ \n\
-- Choose the cosmology                                      \n\
------------------------------------------------------------ \n\
-- Cosmology: LCDM, w0waCDM, DGP, JBD, ...                   \n\
cosmology_model = \""+self.cosmology+"\"                     \n\
cosmology_OmegaCDM = "+str(self.Omega_cdm)+"                 \n\
cosmology_Omegab = "+str(self.Omega_b)+"                     \n\
cosmology_OmegaMNu = "+str(self.Omega_ncdm)+"                \n\
cosmology_OmegaLambda = "+str(self.Omega_fld)+"              \n\
cosmology_Neffective = "+str(self.Neff)+"                    \n\
cosmology_TCMB_kelvin = "+str(self.T_cmb)+"                  \n\
cosmology_h = "+str(self.h)+"                                \n\
cosmology_As = "+str(self.A_s)+"                             \n\
cosmology_ns = "+str(self.n_s)+"                             \n\
cosmology_kpivot_mpc = "+str(self.k_pivot)+"                 \n\
                                                             \n\
-- The w0wa parametrization                                  \n\
if cosmology_model == \"w0waCDM\" then                       \n\
  cosmology_w0 = "+str(self.w0_fld)+"                        \n\
  cosmology_wa = "+str(self.wa_fld)+"                        \n\
end                                                          \n\
                                                             \n\
------------------------------------------------------------ \n\
-- Choose the gravity model                                  \n\
------------------------------------------------------------ \n\
-- Gravity model: GR, DGP, f(R), JBD, Geff, ...              \n\
gravity_model = \""+gravity+"\"                              \n\
                                                             \n\
-- Hu-Sawicky f(R) gravity                                   \n\
if gravity_model == \"f(R)\" then                            \n\
  gravity_model_fofr_fofr0 = "+str(self.fofr0)+"             \n\
  gravity_model_fofr_nfofr = 1.0                             \n\
  gravity_model_fofr_exact_solution = false                  \n\
  gravity_model_screening = "+self.screening+"               \n\
  gravity_model_screening_enforce_largescale_linear = "+self.largscale_linear+" \n\
  gravity_model_screening_linear_scale_hmpc = 0.1            \n\
  gravity_model_screening_efficiency = "+str(self.screening_efficiency)+" \n\
end                                                          \n\
------------------------------------------------------------ \n\
-- Particles                                                 \n\
------------------------------------------------------------ \n\
-- Number of CDM+b particles per dimension                   \n\
particle_Npart_1D = "+str(self.npart)+"                      \n\
-- Factor of how many more particles to allocate space       \n\
particle_allocation_factor = 1.5                             \n\
                                                             \n\
------------------------------------------------------------ \n\
-- Output                                                    \n\
------------------------------------------------------------ \n\
-- List of output redshifts                                  \n\
output_redshifts = {0.0}                                     \n\
-- Output particles?                                         \n\
output_particles = false                                     \n\
-- Fileformat: GADGET, FML                                   \n\
output_fileformat = \"GADGET\"                               \n\
-- Output folder                                             \n\
output_folder = \""+self.outputfolder+"\"                    \n\
                                                             \n\
------------------------------------------------------------ \n\
-- Time-stepping                                             \n\
------------------------------------------------------------ \n\
-- Number of steps between the outputs (in output_redshifts) \n\
timestep_nsteps = {"+str(self.ntimesteps)+"}                 \n\
-- The time-stepping method: Quinn, Tassev                   \n\
timestep_method = \"Quinn\"                                  \n\
-- For Tassev: the nLPT parameter                            \n\
timestep_cola_nLPT = -2.5                                    \n\
-- The time-stepping algorithm: KDK                          \n\
timestep_algorithm = \"KDK\"                                 \n\
-- Spacing of the time-steps in 'a': linear, logarithmic, .. \n\
timestep_scalefactor_spacing = \"linear\"                    \n\
                                                             \n\
------------------------------------------------------------ \n\
-- Initial conditions                                        \n\
------------------------------------------------------------ \n\
-- The random seed                                           \n\
ic_random_seed = "+str(self.seed)+"                          \n\
-- The random generator (GSL or MT19937).                    \n\
ic_random_generator = \"GSL\"                                \n\
-- Fix amplitude when generating the gaussian random field   \n\
ic_fix_amplitude = true                                      \n\
-- Mirror the phases (for amplitude-fixed simulations)       \n\
ic_reverse_phases = false                                    \n\
ic_random_field_type = \"gaussian\"                          \n\
-- The grid-size used to generate the IC                     \n\
ic_nmesh = particle_Npart_1D                                 \n\
-- For MG: input LCDM P(k) and use GR to scale back and      \n\
-- ensure same IC as for LCDM                                \n\
ic_use_gravity_model_GR = "+self.input_spectra_from_lcdm+"   \n\
-- The LPT order to use for the IC                           \n\
ic_LPT_order = 2                                             \n\
-- The type of input:                                        \n\
-- powerspectrum    ([k (h/Mph) , P(k) (Mpc/h)^3)])          \n\
-- transferfunction ([k (h/Mph) , T(k)  Mpc^2)]              \n\
-- transferinfofile (a bunch of T(k,z) files from CAMB)      \n\
ic_type_of_input = \"transferinfofile\"                      \n\
-- When running CLASS we can just ask for outputformat CAMB  \n\
ic_type_of_input_fileformat = \"CAMB\"                       \n\
-- Path to the input                                         \n\
ic_input_filename = \""+colatransferinfofilename+"\"         \n\
-- The redshift of the P(k), T(k) we give as input           \n\
ic_input_redshift = 0.0                                      \n\
-- The initial redshift of the simulation                    \n\
ic_initial_redshift = "+str(self.zini)+"                     \n\
-- Normalize wrt sigma8?                                     \n\
-- If ic_use_gravity_model_GR then this is the sigma8 value  \n\
-- in a corresponding GR universe!                           \n\
ic_sigma8_normalization = "+self.sigma8_norm+"               \n\
ic_sigma8_redshift = 0.0                                     \n\
ic_sigma8 = "+str(self.sigma8)+"                             \n\
                                                             \n\
------------------------------------------------------------ \n\
-- Force calculation                                         \n\
------------------------------------------------------------ \n\
-- Grid to use for computing PM forces                       \n\
force_nmesh = "+str(self.nmesh)+"                            \n\
-- Density assignment method: NGP, CIC, TSC, PCS, PQS        \n\
force_density_assignment_method = \"CIC\"                    \n\
-- The kernel to use when solving the Poisson equation       \n\
force_kernel = \"continuous_greens_function\"                \n\
-- Include the effects of massive neutrinos when computing   \n\
-- the density field (density of mnu is linear prediction    \n\
-- Requires: transferinfofile above (we need all T(k,z))     \n\
force_linear_massive_neutrinos = true                        \n\
                                                             \n\
------------------------------------------------------------ \n\
-- On the fly analysis                                       \n\
------------------------------------------------------------ \n\
                                                             \n\
------------------------------------------------------------ \n\
-- Halofinding                                               \n\
------------------------------------------------------------ \n\
fof = false                                                  \n\
fof_nmin_per_halo = 20                                       \n\
fof_linking_length = 0.2                                     \n\
fof_nmesh_max = 0                                            \n\
fof_buffer_length_mpch = 3.0                                 \n\
                                                             \n\
------------------------------------------------------------ \n\
-- Power-spectrum evaluation                                 \n\
------------------------------------------------------------ \n\
pofk = false                                                 \n\
pofk_nmesh = force_nmesh                                     \n\
pofk_interlacing = true                                      \n\
pofk_subtract_shotnoise = true                               \n\
pofk_density_assignment_method = \"PCS\"                     \n\
                                                             \n\
------------------------------------------------------------ \n\
-- Power-spectrum multipole evaluation                       \n\
------------------------------------------------------------ \n\
pofk_multipole = false                                       \n\
pofk_multipole_nmesh = force_nmesh                           \n\
pofk_multipole_interlacing = true                            \n\
pofk_multipole_subtract_shotnoise = false                    \n\
pofk_multipole_ellmax = 4                                    \n\
pofk_multipole_density_assignment_method = \"PCS\"           \n\
                                                             \n\
------------------------------------------------------------ \n\
-- Bispectrum evaluation                                     \n\
------------------------------------------------------------ \n\
bispectrum = false                                           \n\
bispectrum_nmesh = 128                                       \n\
bispectrum_nbins = 10                                        \n\
bispectrum_interlacing = true                                \n\
bispectrum_subtract_shotnoise = false                        \n\
bispectrum_density_assignment_method = \"PCS\"               \n\
  """
  
    print("Writing COLA inputfile")
    colainputfile = self.outputfolder + "/cola_input_"+self.name+".lua"
    with open(colainputfile, 'w') as f:
        f.write(colafile)
    
    # [5] Compute HMCode pofk from class (useful to have)
    #cosmo.struct_cleanup()
    #cosmo.empty()
    #cosmo = Class()
    #params["non_linear"] = "hmcode"
    #cosmo.set(params)
    #cosmo.compute()
    #for _z in zarr:
    #  karr = transfer['k (h/Mpc)']
    #  pofk_total = np.array([cosmo.pk(_k * self.h, _z)*self.h**3 for _k in karr])
    #  pofk_cb = np.array([cosmo.pk_cb(_k * self.h, _z)*self.h**3 for _k in karr])
    #  pofkfilename = "class_power_hmcode_"+self.name+"_z"+str(_z)+".txt"
    #  np.savetxt(self.outputfolder + "/" + pofkfilename, np.c_[karr, pofk_cb, pofk_total], header="# k (h/Mpc)  P_tot(k)   P_cb(k)  (Mpc/h)^3")
    
    # [6] Clean up CLASS 
    print("Cleaning CLASS")
    cosmo.struct_cleanup()
    cosmo.empty()

    # [7] Print command to run COLA
    print("Adding run-commands to bash script")
    if self.bashfiles is not None:
      nthreads   = self.nthreads
      ncpu       = self.ncpu
      npernode   = self.npernode
      hostfile   = self.hostfile
      node_number= self.node_number
      print("node_number=",node_number,"bashfiles=",self.bashfiles)
      logfile    = self.outputfolder+"/log_cola_"+self.name+".txt"
      pofkfile   = self.outputfolder+"/pofk_"+self.name+"_cb_z0.000.txt"
      bashfile   = self.bashfiles[node_number]
      if not os.path.isfile(bashfile):
        with open(bashfile, "w") as f:
          f.write("#!/bin/bash\n\n")
      runcommand = "# Simulation ["+self.name+"] \n\
logfile={0}\n\
colaexe={1}\n\
colafile={2}\n\
hostfile={3}\n\
pofkfile={4}\n\
if [[ ! -e \"$pofkfile\" ]]; then\n\
  OMP_NUM_THREADS={5} mpirun -np {6} -ppn {7} -hostfile $hostfile $colaexe $colafile 2>&1 | tee -a $logfile\n\
else\n\
  echo Simulation has already been run!\n\
fi\n\n".format(logfile,self.colaexe,colainputfile,hostfile,pofkfile,nthreads,ncpu,npernode)
      if bashfile is not None:
        with open(bashfile, "a") as f:
          f.write(runcommand)
      else:
        print(runcommand)

     # [9] We are done

# Generate input for all sims and for both MG and GR
for key in data:
  run_param = data[key]
  g = Generate(run_param, bashfiles)
  g.generate_data(GR=False)
  g.generate_data(GR=True)

# Make script executable
if generate_bash_files:
  for bashfile in bashfiles:
    subprocess.run(["chmod", "u+x", bashfile])
