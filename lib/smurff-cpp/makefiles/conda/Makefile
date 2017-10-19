ifndef CONDA_PREFIX
        $(error CONDA_PREFIX is undefined)
endif
ROOT=../..
CXXFLAGS+=-O3 -ffast-math -g -fstrict-aliasing 
CXXFLAGS+=-fopenmp 
LIBLOCS=-Wl,-rpath -Wl,$(CONDA_PREFIX)/lib -L$(CONDA_PREFIX)/lib

include $(ROOT)/Makefile.common

install: smurff
	install smurff $(CONDA_PREFIX)/bin

