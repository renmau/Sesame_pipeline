----

# Input data for the emulator

---

This folder contains 3 files: train.csv, test.csv, and val.csv containing the data we want to emulate. The data provided here is for the linear emulator.

Unzip the zipped files test.csv.bz2, train.csv.bz2, and val.csv.bz2 to get simulation data for the unscreened f(R) emulator in the linear boost case in [Mauland et al. 2023](https://arxiv.org/abs/2309.13295).

This can be done by:
bunzip2 test.csv.bz2 train.csv.bz2 val.csv.bz2

The file containing the training data (train.csv) is quite large (around 1G).

This data can be used to make example plots in the Emulators folder and show you how to access the emulators and compare it to the simulation data. 

----

## Scripts to gather simulation data and make input files

----

Here, there are two possibilities, either the generate_data_z_all.py or the generate_data_z_all_separated.py. The first generates the three files if you performed all your samples at once, and the latter if you performed separate Latin hypercube samplings for each data set.

[generate_data_z_all.py](neural_network_emulator/data/generate_data_z_all.py):

- The get_available_redshifts function needs to be pointed to your output folder with the COLASolver data, and the prefix you gave for your data output needs to be added for the file_pattern variable. This function then goes through the files and finds all available redshift outputs for your data.

- In the read_file function you also have to point to your output folder for the simulation data.

- Further down you must edit the .json file parameter to point to your .json file with all the samples, and the tot_samples parameter to match the amount of samples you have.

- The code then assigns simulations randomly to the training, validation, and testing datasets in an approximately 80-10-10 distribution.

[generate_data_z_all_separated.py](neural_network_emulator/data/generate_data_z_all_separated.py):

- The function does the same as for generate_data_z_all.py, but everything is now done individually for the three sets you already made when you made the samples individually. You must change the number of samples in sets_num to fit your distribution, and also the names pointing to folders and files so that they match the structure and naming you made when you made the samples earlier.
