rm -rf build
mkdir build
cd build
cmake ../ -DCMAKE_BUILD_TYPE=Debug -DENABLE_VERBOSE_COMPILER_LOG=ON -DENABLE_BOOST_RANDOM=ON
make
make test

