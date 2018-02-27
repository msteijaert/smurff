rm -rf build
mkdir build
cd build
cmake ../ -DCMAKE_BUILD_TYPE=Release -DENABLE_VERBOSE_COMPILER_LOG=ON
make
make test
