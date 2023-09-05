# Creating the emulator

-----

Here we provide a *minimal* implementation of a neutral network emulator in case you don't have or know how to write your own.

There are many things this pipeline does not do (which might be important)
- We do not scale the parameters and leave this to the user. In general you probably want to rescale the input/output parameters (e.g. OmegaM, w0, boost, etc in such a way that they all lie in [-1,1]). This often makes it easier for the emulator to produce good results. If so then the parameters must also be scaled in the same way when evaluating the emulator.
- We only provide a minimal set of parameters to tune (loss-function, activation-function, etc) and for more complicated setups you might need to tune more than this. You will then have to add support for this yourself.

-----

## Gather data

-----

The data you created by running the simulations should be gathered (for our linear emulator they are located in the [trainingdata](trainingdata/) folder).

The input is .cvs files with one header line with the name of the inputs (features) and outputs (labels):
- paramname1, paramname2, ..., paramnameN, outname1, ..., outnameM
And then all the data follows row by row where each row has the format:
- inputvalue1, inputvalue2, ..., inputvalueN, outputvalue1, ouputvalue2, ..., outputvalueM

There needs to be 3 files like this (and paths to these files are set in [input.yaml](input.yaml)):
- train.csv
- val.csv
- test.csv

The train.csv data is used for training, val.csv for validation, and test.csv for testing. 

-----

## Train the network

-----

Once you have gathered the data you will have to edit the parameter file [input.yaml](input.yaml):
- Give the name of the input (feature_columns) and output (label_columns) parameters (same names as in the header of the .csv files).
- Set parameters: hidden_layers, learning_rate, batch_size, etc.

Once this is set you can run the code as
- python Emulator.py

After it is done the emulator is in emulator/lightning_logs/version_X/ with X corresponding to the Xth run you have done. Every new run of the emulator (e.g. when you change the parameters) will create a new folder. You can clean it up by simply deleting all the emulator/lightning_logs/version_* folders.

Some general tips here:
- Start with a lower number of hidden layers (one or two) and neurons in each layer to avoid overfitting. Increase if the features of the curves are not resolved accurately enough.
- A small batch size typically increases the training time and a larger batch size decreases it. A typical starting point is often batch_size=32. 
-----

## Test the output

-----

The [FinishedEmulators](../../FinishedEmulators) folder contains the emulators from [Mauland-Hus et al. 2023](#) and instructions on how to use it and plot the output against the test.csv data.

-----

## Python modules needed to train the emulator:

-----

- yaml
- torch
- numpy
- pandas
- pytorch_lightning
- pydantic
- typing
- scikit-learn
