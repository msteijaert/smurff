#!/bin/bash

cd lib/macau-cpp/openmp

sed -i -e 's/^LIBLOCS/#LIBLOCS/g' Makefile
sed -i -e 's/^CXX/#CXX/g' Makefile

echo "CXXFLAGS+=-I/usr/loca/include" >> Makefile
echo "CXXFLAGS+=-I${PREFIX}/include/eigen3" >> Makefile
echo "LDFLAGS+=-L${PREFIX}/lib -lopenblas" >> Makefile
echo 'smurff: cmd_nompi.o $(LIBMACAU) cmd_session.o ' >> Makefile
echo '	$(CXX) $(CXXFLAGS) $(OUTPUT_OPTIONS) $? -o smurff  $(LDFLAGS)' >> Makefile


if [ -n "$OSX_ARCH" ]
then
    make -j smurff CXX=g++-6 
else 
    sed -i -e '/argp/d' Makefile
    make -j smurff
fi

install smurff $PREFIX/bin/
