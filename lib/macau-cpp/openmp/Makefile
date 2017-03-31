ROOT=..

CXXFLAGS+=-O3 -ffast-math -g -fstrict-aliasing -DNDEBUG -march=native -flto
#CXXFLAGS+=-O2 -ffast-math -g -fstrict-aliasing
CXXFLAGS+=-fopenmp
#CXXFLAGS+=-pg
LIBLOCS+=-Wl,-rpath,$(EBROOTGCC)/lib64
LIBLOCS+=-Wl,-rpath,$(EBROOTOPENBLAS)/lib -lgfortran

include ../Makefile.common
