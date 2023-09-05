# The Simulation code

The FML folder contains a fork of the full FML library. You can download the latest version [here](https://github.com/HAWinther/FML/) (just note that if you do then the parameterfile might have changed slightly and in that case you will have to update the script that generate COLA inputs). 

For the Sesame pipeline we only need the COLA solver that is located in [SimulationCode/FML/FML/COLASolver](SimulationCode/FML/FML/COLASolver). The main thing you need to use this code is the FFTW, GSL and LUA library (see below). Go to that folder, edit the [Makefile](SimulationCode/FML/FML/COLASolver/Makefile) with paths to the library (and to FML itself via FML_INCLUDE), compile and run a test to see that it works.

------

## Requirements

The simulation code requires a C++14 (C++17 is reccomended; -std=c++1z so gcc 7, clang 3.9, icpc 19, MSVC 19.11) compatible compiler to compile the library. In addition it requires the following libraries:

 - [FFTW](http://www.fftw.org/download.html) version 3+ 
 - [GSL](https://www.gnu.org/software/gsl/) version 2+ 
 - [LUA](https://www.lua.org/download.html) version 5+ 
