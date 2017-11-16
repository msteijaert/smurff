#!/bin/bash

cd lib/smurff-cpp/makefiles/linux_omp

sed -i -e 's/^LIBLOCS/#LIBLOCS/g' Makefile

echo "CXXFLAGS+=-O3 -ffast-math -g -fstrict-aliasing -DNDEBUG" >> Makefile
echo "CXXFLAGS+=-fopenmp" >> Makefile
echo "CXXFLAGS+=-I${PREFIX}/include/eigen3" >> Makefile
echo "LDFLAGS+=-L${PREFIX}/lib -lopenblas" >> Makefile

[ -n "$OSX_ARCH" ] && {
    echo "CXXFLAGS+=-I/usr/local/include" >> Makefile
    echo "LDFLAGS+=-L/usr/local/lib -largp" >> Makefile
}

make -j smurff

install smurff $PREFIX/bin/
