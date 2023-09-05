# <a name="top"></a>How to use the Pipeline:


The pipeline consists of the following steps that you need to do:
1. [Compile the COLASolver](#compile) in the FML library. 
2. [Generate the Latin hypercube sampling](#genhcs). You will have to determine and set which parameters you want to vary and their priors. Running the script leaves you with a dictionary with the parameter values for all the simulations to be run.
3. [Generate the COLA input](#geninput). You will here read the hypercube sampling dictionary containing the cosmological parameters and the COLA parameters. This generates all the input needed for COLA and the parameter file (also linear data by e.g. running CLASS).
4. [Run the COLA simulations](#runcola)
5. [Make the emulator](#makeemulator)

All parameters now follow a f(R) vs. w0waCDM example as in [Mauland-Hus et al. 2023](#).

---------------------------------------------------

## <a name="compile"></a>1. Compile the COLASolver:

Go to [../SimulationCode/FML/FML/COLASolver](../SimulationCode/FML/FML/COLASolver) and edit the Makefile. Here you will have to set the FML_INCLUDE path (the path to the FML/ folder), the MPI compiler you use, and the paths to the libraries needed by the code (FFTW3, lua, GSL). Compile and test it by running ./nbody parameterfile.lua to check that it runs. Once this is done you have the code setup to run the sims. The executable you will need to run the code is called ./nbody

---------------------------------------------------
## <a name="genhcs"></a>2. Generate the Latin hypercube sampling

This is done by the script [generate_hypercube_sampling.py](generate_hypercube_sampling.py). This script generates a .json file that holds all the different parameter samples (based on Latin hypercube sampling) and simulation specifics for the COLASolver runs. 
You need to specify the number of samples that you want (total_samples), the prefix name you want for the output data when you run the simulations later (prefix), and the name of the .json file (dictfile). In addition, the current setup is set to function by dividing the calculations on several nodes using a hostfile. To use this you put make_hostfiles=True and provide a list of the host names with the hostfiles parameter. Instead of creating all the samples, there is also an option to just plot the samples by putting test_run = True. 

After this, you choose which parameters to vary and their priors (parameters_to_vary), followed by the output folder (outputfolder) (the folder needs to exist as an empty folder before you run this code), and a link to the COLASolver executable (colaexe).

Then you specify the simulation setup for the COLASolver, like the box size, number of particles, meshsize, and so on. This will go into the creation of the COLASolver parameter files. 

Following this you set up the fiducial cosmology (cosmo_param). The parameters that are not varied will follow this value. The ones that are varied will have this value replaced for each sample. These parameters, along with the ones specified for the simulation setup are the same ones as described in the COLASolver folder in the FML library. In addition, there is a choice to use screening and to change the fudge factor, gamma, (screening_efficiency) for the specific f(R) example that this setup follows.

After this, you specify the number of CPUs and CPUs per node, which will be important once the simulations are ready to run.

Run this code: 
- python generate_hypercube_sampling.py
Output: 
- Dictionary: yourfilename.json 

---------------------------------------------------

## <a name="geninput"></a>3. Generate COLA input files

This is done by the script [generate_input.py](generate_input.py). Once generate_hypercube_sampling.py has been run in the previous step, you will have yourfilename.json, a dictionary containing the COLASolver setup information for each parameter sample. This is needed as input for the generate_input code.

You can then run this code by:
python generate_input.py yourfilename.json runfile_prefix num_nodes

Here, runfile_prefix is the name of the .sh files created for initiating the COLASolver simulations for all the samples. There will be one .sh file per node. num_nodes is the number of nodes you added in the hostfiles list in generate_hypercube_sampling.py.

Output: One .sh file per node, which should be initiated at its prescribed node. Say we have hostfiles = ['node1', 'node2'], then we will have runfile_prefix0.sh and runfile_prefix1.sh, where the first should be run at node1 and the second at node2. 

---------------------------------------------------

## <a name="runcola"></a>4. Run the COLA simulations:

Once the run-files from the previous step are generated; run them. The COLASolver will run and the output will be saved in the output folder you linked in generate_hypercube_sampling.py. The code automatically runs one GR run and one with modified gravity.

---------------------------------------------------

## <a name="makeemulator"></a>5. Make the emulator:

Go to the [neural_network_emulator](neural_network_emulator/) folder. This folder contains everything you need to take the data from the COLASolver simulations and make them into an emulator. 

Some general info:
- With the codes provided in this folder, you can take two different approaches to making all the samples and running all the simulations.
You can either do the above process three times, creating samples corresponding to an 80-10-10 percent distribution to create training, test, and validation sets where each of the samples are drawn using Latin hypercube sampling. This ensures an even distribution in parameter space for the three samples. In [Mauland-Hus et al. 2023](#), this corresponded to making one training sample draw with 440 samples, one testing sample draw with 55 samples, and one validation sample draw, with 55 samples, giving a total of 550 samples and an 80-10-10 distribution of the training, testing, and validation samples. 
- An easier approach is to just make 550 samples directly, and then draw the training, testing, and validation samples randomly afterward, corresponding to an 80-10-10 distribution. 


The emulation process consists of three main steps:
1. [Gather the simulation data](#gatherdata)
2. [Train the network](#train)
3. [View the results / use the emulator](#use)

---------------------------------------------------

### <a name="gatherdata"></a>5.1 Make the emulator - Gather the simulation data

---------------------------------------------------

The first thing that needs to be done is to convert the simulation output into training, testing, and validation files in .csv format. Script to do this are located in the [neural_network_emulator/trainingdata/](neural_network_emulator/trainingdata/) folder. Here, there are two possibilities, either the generate_data_z_all.py or the generate_data_z_all_separated.py. The first generates the three files if you performed all your samples at once, and the latter if you performed separate Latin hypercube samplings for each data set. 

[generate_data_z_all.py](neural_network_emulator/trainingdata/generate_data_z_all.py): 

- The get_available_redshifts function needs to be pointed to your output folder with the COLASolver data, and the prefix you gave for your data output needs to be added for the file_pattern variable. This function then goes through the files and finds all available redshift outputs for your data. 

- In the read_file function you also have to point to your output folder for the simulation data.

- Further down you must edit the .json file parameter to point to your .json file with all the samples, and the tot_samples parameter to match the amount of samples you have. 

- The code then assigns simulations randomly to the training, validation, and testing datasets in an approximately 80-10-10 distribution.

[generate_data_z_all_separated.py](neural_network_emulator/trainingdata/generate_data_z_all_separated.py):

- The function does the same as for generate_data_z_all.py, but everything is now done individually for the three sets you already made when you made the samples individually. You must change the number of samples in sets_num to fit your distribution, and also the names pointing to folders and files so that they match the structure and naming you made when you made the samples earlier. 

- In the trainingdata folder, there are the lin_separatedLHS_test.csv example files, which match the ones used to train the fully linear $f(R)$ emulator in [Mauland-Hus et al. 2023](#). 

---------------------------------------------------

### <a name="train"></a>5.2 Make the emulator - Train the network

---------------------------------------------------

Back in the main neural_network_emulator there is only one file we need to change to run the neural network training once the data files are made. That is the [input.yaml](neural_network_emulator/input.yaml) file.

Here you make sure that your data file structure follows that in feature_columns and label_columns, point to the correct data files, create the architecture that you want, and point to the emulators output folder. 

Now you start the training by running:
- python [Emulator.py](neural_network_emulator/Emulator.py)

Output:
- Ends up in emulators/lightning_logs in a folder named version_x where once you rerun the training with a different architecture a new folder with a higher version number is made, so that you do not overwrite your results. You can see how to use the resulting emulator in the Emulators folder containing the $f(R)$ emulators in [Mauland-Hus et al. 2023](#).

---------------------------------------------------

### <a name="use"></a>5.3 View the results / use the emulator

---------------------------------------------------

In the folder [FinishedEmulators](../FinishedEmulators) you find the three emulators created in [Mauland-Hus et al. 2023](#).
Here you also find three files demonstrating how to use an emulator and compare its output to data:

- [example_emulator_usage.py](https://github.com/renmau/Sesame_pipeline/blob/main/FinishedEmulators/example_emulator_usage.py): A simple example of how to use the emulator with data.
  
- [plot_train_test_val_performance.py](https://github.com/renmau/Sesame_pipeline/blob/main/FinishedEmulators/plot_train_test_val_performance.py): Plots the emulator result versus the training, validation, and test data sets used to train it. To use this code, we also provide the data sets used to make the emulator in the unscreened linear $P(k)$ case in [Mauland-Hus et al. 2023](#). To use the data, you must first go to the [trainingdata](https://github.com/renmau/Sesame_pipeline/blob/main/Pipeline/neural_network_emulator/trainingdata) folder and unzip the files.
  
- [plot_test_performance_redshifts.py](https://github.com/renmau/Sesame_pipeline/blob/main/FinishedEmulators/plot_test_performance_redshifts.py): Plots the emulator result versus the test set for three different redshifts. The example data provided for this code is the same as for the point above.

   
