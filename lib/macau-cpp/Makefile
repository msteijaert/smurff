CXX=mpicxx
#CXX=g++
CXXFLAGS=-std=c++11 -Wall -Wno-unknown-pragmas 
#CXXFLAGS+=-fopenmp
CXXFLAGS += -I ../eigen3 -I ../libfastsparse
LIBLOCS := -L/usr/local/Cellar/openblas/0.2.18_2/lib
LDFLAGS := $(LIBLOCS) -lm -lopenblas -lpthread

CXXFLAGS+=-O3 -ffast-math -g -fstrict-aliasing -DNDEBUG -march=native -flto
#CXXFLAGS+=-O3 -ffast-math -g -fstrict-aliasing -flto
#CXXFLAGS+=-O0 -g

LINK.o=$(CXX) $(CXXFLAGS) $(LDFLAGS)
OUTPUT_OPTION=-MMD -MP -o $@

.PHONY: all clean 

all: libmacau.a tests macau_mpi

LIBOBJS = mvnormal.o chol.o linop.o latentprior.o noisemodels.o macau.o macau_mpi.o macauoneprior.o gen_random.o
libmacau.a: $(LIBOBJS)
	$(AR) csr $@ $+
	
test_dense tests macau_mpi: libmacau.a

clean:
	rm -f */*.o *.o */*.d *.d
	rm -f macau_mpi tests libmacau.a
 
# DFILES 
DFILES=$(wildcard */*.d *.d)

-include $(DFILES)
