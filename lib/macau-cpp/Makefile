CXX=g++
CXXFLAGS=-std=c++11 -Wall -Wno-unknown-pragmas -O3 -fstrict-aliasing
#CXXFLAGS+=-fopenmp
LINK.o=$(CXX) $(CXXFLAGS)

INCLUDES := -I ../eigen3 -I ../libfastsparse
LIBLOCS := -L/usr/local//Cellar/openblas/0.2.18_2/lib
LDFLAGS := -lm -lopenblas -lpthread

all: test
test: tests

%.o: %.cpp %.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@
%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

OBJS = mvnormal.o chol.o linop.o latentprior.o tests.o

tests: $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o tests $(OBJS) $(LDFLAGS) $(LIBLOCS)

clean:
	rm -f tests *.o
