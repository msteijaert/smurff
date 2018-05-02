:: Use python version to select which Visual Studio to use
:: For win-64, we'll need more, since those are separate compilers
:: Build in subdirectory.
mkdir build
cd build
if errorlevel 1 exit 1

cmake -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=%LIBRARY_PREFIX% -DCMAKE_PREFIX_PATH=%LIBRARY_PREFIX% ..\lib\smurff-cpp\cmake
if errorlevel 1 exit 1
cmake --build . --target INSTALL --config Release
if errorlevel 1 exit 1

cd python\Smurff
if errorlevel 1 exit 1
%PYTHON% setup.py install --single-version-externally-managed --record=record.txt
if errorlevel 1 exit 1

