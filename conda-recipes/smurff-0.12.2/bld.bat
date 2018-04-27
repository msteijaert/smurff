:: Use python version to select which Visual Studio to use
:: For win-64, we'll need more, since those are separate compilers
:: Build in subdirectory.
mkdir build
cd build

cmake -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=%LIBRARY_PREFIX% -DCMAKE_PREFIX_PATH=%LIBRARY_PREFIX% ..\lib\smurff-cpp\cmake
cmake --build . --target INSTALL --config Release

cd python\Smurff
%PYTHON% setup.py install --single-version-externally-managed --record=record.txt

