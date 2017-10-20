rm build -r
mkdir build
cd build
cmake ../ -DENABLE_OPENBLAS=ON -DCMAKE_BUILD_TYPE=Release
make
../_output/tests
