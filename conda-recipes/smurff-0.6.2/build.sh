#!/bin/bash

eigen_dest="lib/eigen3"
eigen_url="http://bitbucket.org/eigen/eigen/get/3.3.3.tar.bz2"
eigen_inner="eigen-eigen-67e894c6cd8f"

wget -O - $eigen_url | tar xjf -
mv $eigen_inner $eigen_dest

cd lib/macau-cpp/openmp

sed -i -e 's/^LIBLOCS/#LIBLOCS/g' Makefile
sed -i -e 's/^CXX/#CXX/g' Makefile

echo "CXXFLAGS+=-I/usr/local/include" >> Makefile
echo "CXXFLAGS+=-O3 -ffast-math -g -fstrict-aliasing -DNDEBUG" >> Makefile
echo "CXXFLAGS+=-fopenmp" >> Makefile
echo "LDFLAGS+=-L${PREFIX}/lib -lopenblas" >> Makefile
echo 'smurff: cmd_nompi.o $(LIBMACAU) cmd_session.o ' >> Makefile
echo '	$(CXX) $(CXXFLAGS) $(OUTPUT_OPTIONS) $? -o smurff  $(LDFLAGS)' >> Makefile

if [ -n "$OSX_ARCH" ]
then
    make -j smurff
else 
    sed -i -e '/argp/d' Makefile
    make -j smurff
fi

install smurff $PREFIX/bin/
