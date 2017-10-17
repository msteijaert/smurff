ifndef CONDA_PREFIX
        $(error CONDA_PREFIX is undefined)
endif
ROOT=../..
CXXFLAGS+=-O3 -ffast-math -g -fstrict-aliasing -DNDEBUG 
CXXFLAGS+=-fopenmp 
CXXFLAGS=-I/usr/local/include 
LIBLOCS=-Wl,-rpath -Wl,$(CONDA_PREFIX)/lib -L$(CONDA_PREFIX)/lib /usr/local/lib/libargp.a

include $(ROOT)/Makefile.common

install: smurff
	install smurff $(CONDA_PREFIX)/bin

