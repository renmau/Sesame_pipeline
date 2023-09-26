# Sesame 1.0

This is Sesame ([Mauland et al. 2023](https://arxiv.org/abs/2309.13295)) - a fully working pipeline for predicting the non-linear matter-power spectrum in beyond $\Lambda$-CDM models. The pipeline uses the COLA method to perform simulations and machine-learning tools to train an emulator from the simulation data.

The simulation code used is the COLASolver found in the [FML library](https://github.com/HAWinther/FML).

The neural network training is performed using [PyTorch Lightening](https://lightning.ai/), a light-weight wrapper for the [Pytorch module](https://pytorch.org/). 

This repository consists of three folders:

## [SimulationCode](SimulationCode/)
- Contains the FML library version used to build our pipeline. Inside the folder [SimulationCode/FML/FML/COLASolver](SimulationCode/FML/FML/COLASolver) you find the code used to perform simulations for the Sesame pipeline.

## [Pipeline](Pipeline/)
This folder contains the code needed to draw the parameter samples using Latin hypercube sampling, create runfiles for the COLASolver code for all the samples, convert the COLASolver output data to a format suitable for the neural network training, the input file for the neural network training, and the training code itself. 

## [FinishedEmulators](FinishedEmulators/)
This folder contains the three example emulators created as a demonstration of the pipeline in [Mauland et al. 2023](https://arxiv.org/abs/2309.13295).

The Pipeline and Emulators folders contain information files on how to use or access the code and data in the folders.

## Contributors

- Renate Mauland
- Hans A. Winther
- Cheng-Zong Ruan
