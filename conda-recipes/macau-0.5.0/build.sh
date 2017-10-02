sed -i -e 's/library_dirs = ldirs/library_dirs = []/g' setup.py
sed -i -e 's#, "/usr/local/include", "/usr/local/opt/openblas/include"##g' setup.py
sed -i -e "s#'-fopenmp'#'-fopenmp', '-DEIGEN_DONT_PARALLELIZE'#g" setup.py
$PYTHON setup.py install     # Python command to install the script.
