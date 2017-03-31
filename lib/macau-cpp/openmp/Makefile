ROOT=..

CXXFLAGS+=-O3 -ffast-math -g -fstrict-aliasing -DNDEBUG -march=native -flto
#CXXFLAGS+=-O2 -ffast-math -g -fstrict-aliasing
CXXFLAGS+=-fopenmp
#CXXFLAGS+=-pg
CXXFLAGS+=-Wl,-rpath,$(EBROOTGCC)/lib64
CXXFLAGS+=-Wl,-rpath,$(EBROOTOPENBLAS)/lib

include ../Makefile.common
