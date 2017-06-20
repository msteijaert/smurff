ROOT=../..

CXXFLAGS+=-O3 -ffast-math -g -fstrict-aliasing -DNDEBUG -march=native -flto -fopenmp
#CXXFLAGS+=-O2 -g 
#CXXFLAGS+=-O0 -g

CLANGROOT=/usr/local/Cellar/llvm/4.0.0_1
LIBLOCS+=-Wl,-rpath,$(CLANGROOT)/lib
LIBLOCS+=-L$(CLANGROOT)/lib

include ../Makefile.mac

CXX=$(CLANGROOT)/bin/clang++

