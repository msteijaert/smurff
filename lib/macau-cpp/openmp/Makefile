ROOT=..

CXXFLAGS+=-O3 -ffast-math -g -fstrict-aliasing -DNDEBUG -march=native 
#CXXFLAGS+=-O2 -ffast-math -g -fstrict-aliasing
CXXFLAGS+=-fopenmp
CXXFLAGS+=-DEIGEN_DONT_PARALLELIZE
#CXXFLAGS+=-pg
#LIBLOCS+=-Wl,-rpath,$(EBROOTGCC)/lib64
#LIBLOCS+=-Wl,-rpath,$(EBROOTOPENBLAS)/lib -lgfortran
LIBLOCS+=-L/usr/local/Cellar/openblas/0.2.19_1/lib

include ../Makefile.common

# for intel
#LDFLAGS := $(LIBLOCS) -lmkl
#CXX = icpc

#mac
LDFLAGS+=-largp
CXX=g++-6
