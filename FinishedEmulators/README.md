# Completed Emulators

The three emulators created for $f(R)$ gravity which were used as an exmaple in [Mauland et al. 2023](https://arxiv.org/abs/2309.13295) are given in folders:
- [screened_fofr_nonlin_Pk](FinishedEmulators/screened_fofr_nonlin_Pk/version_7/) non-linear $f(R)$ with screening
- [unscreened_fofr_nonlin_Pk](FinishedEmulators/unscreened_fofr_nonlin_Pk/version_4/) non-linear $f(R)$ without screening
- [unscreened_fofr_lin_Pk](FinishedEmulators/unscreened_fofr_lin_Pk/version_3/) $f(R)$ using linear theory

They emulate the boost between $f(R)$ and GR in the case of screened $f(R)$ and non-lin $P(k)$, unscreened $f(R)$ and linear $P(k)$, and unscreened $f(R)$ and non-lin $P(k)$.

## Python modules needed to use the emulator:

- yaml
- torch
- numpy
- pandas
- pytorch_lightning
- pydantic
- typing
- scikit-learn

## Example of using emulator - [example_emulator_usage.py](example_emulator_usage.py)
This script shows a minimal example of how to call the three emulator(s) here and get a prediction back. 

There are also two codes that show how to access and plot the emulator results in this folder. 

## Plot code - [plot_train_test_val_performance.py](plot_train_test_val_performance.py):
Plots the training, testing, and validation data samples against the emulator prediction for the unscreened linear case at z=0.0. 

## Plot code - [plot_test_performance_redshifts.py](plot_test_performance_redshifts.py):
Plots the test data sample against the emulator predictions for three different redshifts for the unscreened linear case.

Comment on the plot code:
- The data in the .csv files (train, test, val) are here saved in one long continuous data stream, one $k$ and $P(k)$ value after the other. To plot them individually we need to separate them, which is done through a split parameter. In both codes, this is set to split=384. This is the specific length of the $k$-array for the simulations performed in [Mauland et al. 2023](https://arxiv.org/abs/2309.13295). This needs to be swapped out with the length of the array of your own simulations.

