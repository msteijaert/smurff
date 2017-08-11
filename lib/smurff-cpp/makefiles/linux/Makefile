ROOT=../../..

CXX=g++
CXXFLAGS=-std=c++11 -Wall -Wno-unknown-pragmas -fPIC
INCLUDES=-I $(ROOT)/eigen3 -I $(ROOT)/libfastsparse
LDFLAGS=-lm -lopenblas -lpthread

LIBOBJS = mvnormal.o chol.o linop.o noisemodels.o session.o gen_random.o model.o result.o
LIBOBJS += matrix_io.o macauoneprior.o spikeandslab.o macauprior.o normalprior.o data.o
LIBOBJS += ini.o INIReader.o
LIBOBJS += counters.o
LIBSMURFF = libsmurff-cpp.a

.PHONY: all clean smurff tests

all: smurff

clean:
	rm -f *.o *.a smurff tests

smurff: $(LIBSMURFF) cmd_session.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(ROOT)/smurff-cpp/smurff.cpp $(LIBSMURFF) $(LDFLAGS) cmd_session.o -o $@

tests: $(LIBSMURFF)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(ROOT)/smurff-cpp/tests.cpp $(LIBSMURFF) $(LDFLAGS) -o $@

$(LIBSMURFF): $(LIBOBJS)
	ar -rcs $(LIBSMURFF) $(LIBOBJS)

%.o : $(ROOT)/smurff-cpp/%.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

%.o : $(ROOT)/smurff-cpp/%.c
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@
