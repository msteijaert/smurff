ifndef CONDA_PREFIX
        $(error CONDA_PREFIX is undefined)
endif
ROOT=../..
CXXFLAGS+=-O3 -ffast-math -g -fstrict-aliasing -DNDEBUG -flto
CXXFLAGS+=-fopenmp 
LIBLOCS=-Wl,-rpath=\$$ORIGIN/../lib -L$(CONDA_PREFIX)/lib

include $(ROOT)/Makefile.common

install: smurff
	install smurff $(CONDA_PREFIX)/bin

