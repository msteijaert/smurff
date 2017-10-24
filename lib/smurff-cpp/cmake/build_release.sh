rm -rf build
mkdir build
cd build
cmake ../ -DENABLE_OPENBLAS=ON -DCMAKE_BUILD_TYPE=Release
make
Release/tests
