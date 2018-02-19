sed -i -e 's/library_dirs = ldirs/library_dirs = []/g' setup.py
sed -i -e 's#, "/usr/local/include", "/usr/local/opt/openblas/include"##g' setup.py
sed -i -e "s#'-fopenmp'#'-fopenmp', '-DEIGEN_DONT_PARALLELIZE'#g" setup.py
sed -i -e 's#blas_libs = get_blas_libs()#blas_libs = ["openblas"]#g' setup.py
$PYTHON setup.py install  --single-version-externally-managed --record=record.txt
