CXX=g++
CXXFLAGS=-std=c++11 -Wall -Wno-unknown-pragmas -O0 -fstrict-aliasing
#CXXFLAGS+=-fopenmp
LINK.o=$(CXX) $(CXXFLAGS)

INCLUDES := -I ../eigen3 -I ../libfastsparse
LIBLOCS := -L/usr/local/Cellar/openblas/0.2.18_2/lib
LDFLAGS := -lm -lopenblas -lpthread

all: test macau_mpi
test: tests 

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

OBJS = mvnormal.o chol.o linop.o latentprior.o noisemodels.o macau.o macau_mpi.o macauoneprior.o

tests: $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o tests $(OBJS) $(LDFLAGS) $(LIBLOCS)

macau_mpi: ${OBJS}
	${MPICXX} ${CXXFLAGS} ${INCLUDES} -o macau_mpi ${OBJS} ${LDFLAGS} ${LIBLOCS}

clean:
	rm -f macau_mpi tests *.o
