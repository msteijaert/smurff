ROOT=../../..

CXX=g++
CXXFLAGS=-std=c++11 -Wall -Wno-unknown-pragmas -fPIC
INCLUDES=-I $(ROOT)/eigen3 -I $(ROOT)/libfastsparse
LDFLAGS=-lm -lopenblas -lpthread

LIBOBJS = inv_norm_cdf.o truncnorm.o mvnormal.o chol.o linop.o noisemodels.o session.o gen_random.o model.o result.o
LIBOBJS += matrix_io.o macauoneprior.o spikeandslab.o macauprior.o normalprior.o data.o
LIBOBJS += ini.o INIReader.o
LIBOBJS += counters.o
LIBSMURFF = libsmurff-cpp.a

.PHONY: all clean smurff tests bench_nrandn bench_wishart bench_mvnormal

all: smurff

clean:
	rm -f *.o *.a smurff tests bench_nrandn bench_wishart bench_mvnormal

smurff: $(LIBSMURFF) cmd_session.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(ROOT)/smurff-cpp/smurff.cpp $(LIBSMURFF) $(LDFLAGS) cmd_session.o -o $@

tests: $(LIBSMURFF)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(ROOT)/smurff-cpp/tests.cpp $(LIBSMURFF) $(LDFLAGS) -o $@

bench_nrandn:
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(ROOT)/smurff-cpp/mvnormal.cpp $(LDFLAGS) -DBENCH -DBENCH_NRANDN -o $@

bench_wishart:
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(ROOT)/smurff-cpp/mvnormal.cpp $(LDFLAGS) -DBENCH -DBENCH_COND_NORMALWISHART -DBENCH_NORMAL_WISHART -o $@

bench_mvnormal:
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(ROOT)/smurff-cpp/mvnormal.cpp $(LDFLAGS) -DBENCH -DBENCH_MVNORMAL -o $@

$(LIBSMURFF): $(LIBOBJS)
	ar -rcs $(LIBSMURFF) $(LIBOBJS)

%.o : $(ROOT)/smurff-cpp/%.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

%.o : $(ROOT)/smurff-cpp/%.c
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

