# Hans A. Winther (2015) (hans.a.winther@gmail.com)

SHELL := /bin/bash

#===================================================
# Set c++11 compliant compiler. If USE_MPI we use MPICC 
#===================================================

CC      = g++ -std=c++1z -O3 -Wall -Wextra -march=native
MPICC   = mpicxx -std=c++1z -O3 -Wall -Wextra -march=native

#===================================================
# Options
#===================================================

# Use MPI
USE_MPI          = false
# Use OpenMP threads
USE_OMP          = true
# Use the GSL library (Spline, ODESolver)
USE_GSL          = true
# Check for bad memory accesses
USE_SANITIZER    = false

#===================================================
# Include and library paths
#===================================================

# Main library include (path to folder containin FML/)
FML_INCLUDE    = $(HOME)/local/FML

# GSL : only needed if USE_GSL = true
GSL_INCLUDE    = $(HOME)/local/include
GSL_LIB        = $(HOME)/local/lib
GSL_LINK       = -lgsl -lgslcblas

#===================================================
# Compile up all library defines from options above
#===================================================

INC     = -I$(FML_INCLUDE) 
LIB     =
LINK    = 
OPTIONS = -DHMCODETESTING

ifeq ($(USE_DEBUG),true)
OPTIONS += 
endif

ifeq ($(USE_MPI),true)
CC       = $(MPICC)
OPTIONS += -DUSE_MPI
endif

ifeq ($(USE_OMP),true)
OPTIONS += -DUSE_OMP
CC      += -fopenmp
endif

ifeq ($(USE_SANITIZER),true)
CC      += -fsanitize=address
endif

ifeq ($(USE_GSL),true)
OPTIONS += -DUSE_GSL
INC     += -I$(GSL_INCLUDE)
LIB     += -L$(GSL_LIB)
LINK    += $(GSL_LINK)
endif

VPATH := $(FML_INCLUDE)/FML/Global/:$(FML_INCLUDE)/FML/ODESolver/:$(FML_INCLUDE)/FML/Math/:$(FML_INCLUDE)/FML/Spline/:$(FML_INCLUDE)/FML/HaloModel/:$(FML_INCLUDE)/FML/SphericalCollapse/: $(FML_INCLUDE)/FML/FileUtils
OBJS = Main.o Global.o Math.o Spline.o ODESolver.o SphericalCollapse.o SphericalCollapseModel.o Halomodel.o FileUtils.o

TARGETS := halomodel
all: $(TARGETS)
.PHONY: all clean

clean:
	rm -rf $(TARGETS) *.o

halomodel: $(OBJS)
	${CC} -o $@ $^ $(OPTIONS) $(LIB) $(LINK)

%.o: %.cpp 
	${CC} -c -o $@ $< $(OPTIONS) $(INC) 

