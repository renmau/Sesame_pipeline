import numpy as np
import matplotlib.pyplot as plt
from Emulator import EmulatorEvaluator

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
# Paths to emulator(s)
#=============================================
lin_emulator_folder = "./unscreened_fofr_lin_Pk"
lin_emulator_version = 0
lin_emulator_function = PofkBoostEmulator(path = lin_emulator_folder, version = lin_emulator_version)

nl_noscreening_emulator_folder = "./unscreened_fofr_nonlin_Pk/"
nl_noscreening_emulator_version = 7
nl_noscreening_emulator_function = PofkBoostEmulator(path = nl_noscreening_emulator_folder, version = nl_noscreening_emulator_version)

nl_screening_emulator_folder = "./screened_fofr_nonlin_Pk/"
nl_screening_emulator_version = 4
nl_screening_emulator_function = PofkBoostEmulator(path = nl_screening_emulator_folder, version = nl_screening_emulator_version)

#=============================================
# The cosmological parameters
#=============================================
sigma8_GR = 0.8
Omegacdm  = 0.25
w0        = -1.0
wa        = 0.0
log10fofr = -6.0
z         = 0.0

#=============================================
# The k-array we want the results for
#=============================================
khMpc_array = np.linspace(1e-2, 1.0, 50)

#=============================================
# Parameters to be given to emulator
#=============================================
params_cosmo = np.array([
    sigma8_GR,
    Omegacdm,    
    w0,
    wa,
    log10fofr,
    z])
params_batch = np.column_stack(( np.vstack([params_cosmo] * len(khMpc_array)), np.log10(khMpc_array)))

#=============================================
# Call the emulators
#=============================================
lin_boost            = lin_emulator_function(params_batch)
nl_screening_boost   = nl_screening_emulator_function(params_batch)
nl_noscreening_boost = nl_noscreening_emulator_function(params_batch)

#=============================================
# Plot the results for P(k) / PGR(k)
#=============================================
plt.xlabel("k (h/Mpc)")
plt.ylabel("B(k,z=0)")
plt.xscale("log")
plt.plot(khMpc_array, lin_boost, label="Linear")
plt.plot(khMpc_array, nl_noscreening_boost, label="Non-linear (no screening)")
plt.plot(khMpc_array, nl_screening_boost, label="Non-linear (with screening)")
plt.legend()
plt.show()
