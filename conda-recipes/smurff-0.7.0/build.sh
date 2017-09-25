#!/bin/bash

cd lib/smurff-cpp/makefiles/linux_omp

sed -i -e 's/^LIBLOCS/#LIBLOCS/g' Makefile
sed -i -e 's/^CXX/#CXX/g' Makefile

echo "CXXFLAGS+=-I${PREFIX}/include/eigen3" >> Makefile
echo "LDFLAGS+=-L${PREFIX}/lib -lopenblas" >> Makefile

[ -n "$OSX_ARCH" ] && {
    echo "CXXFLAGS+=-I/usr/local/include" >> Makefile
    echo "LDFLAGS+=-L/usr/local/lib -largp" >> Makefile
}

make -j smurff

install smurff $PREFIX/bin/
