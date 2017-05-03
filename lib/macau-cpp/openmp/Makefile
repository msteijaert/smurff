ROOT=..

#CXXFLAGS+=-O3 -ffast-math -g -fstrict-aliasing -DNDEBUG -march=native -flto
CXXFLAGS+=-O3 -ffast-math -g -fstrict-aliasing -DNDEBUG -march=native 
#CXXFLAGS+=-O2 -ffast-math -g -fstrict-aliasing
CXXFLAGS+=-fopenmp
CXXFLAGS+=-DEIGEN_DONT_PARALLELIZE
#CXXFLAGS+=-pg

include ../Makefile.common

# for intel
#LDFLAGS := $(LIBLOCS) -lmkl
#CXX = icpc

#mac
LDFLAGS+=-largp
CXX=g++-6
