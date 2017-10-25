Brainy SMURFF Test Suite
========================

This directory contains python scripts to run the SMURFF C++ command line tool
on several data sets and with several command options, and to verify the
results.



## Requirements

The test suite scripts are written in Python 3. A recent version of Python 3 is
needed, along with these packages: 

* scipy
* pandas
* scikit-learn
* matrix_io (available on anaconda, and in ../matrix_io)

## Running

To run the suite, these are the steps:


### Step 1: Install one or more smurff versions

For each SMURFF version installed there should be a conda evironment. The bash
script ``install_conda_envs.sh`` installs several smurff versions from anaconda.org
in ``conda_envs/``

### Step 2: Generate the data files

There are currenly tests for three datasets:

1. *synthetic data*: first a random model is generated, next a dataset is
   generated from this model. Run `make.py` in `data/synthetic` to generate this
   datset.

2. *movielens data*: Movie rating data from http://www.grouplens.org. Run
   `make.py` in `data/movielens` to download and preprocess the movielens
   datset with 100k ratings.

3. *chembl data*: A subset of the ChEMBL (https://www.ebi.ac.uk/chembl/) dataset
   containing chemical compound on protein target bioactivity. A download link
   to this dataset will be provided later.

### Step 3: Generate the tests.

Run `./gen_tests.py`. 

This will generate a `cmd` file for each test in a *work* directory (`work/` by
default), for each conda environment. A link to the latest called `latest` set
of generated tests will be in `work/`:


### Step 4: Run the tests.

Tests are run by execution the `cmd` `bash` script in every `work/latest` subdirectory. For example:

* Executing one test:
  `$ bash -e ./smurff-0.6.1/`...`/cmd`
* Using `find` and `xargs`:
  `$ find . -name cmd | xargs -n 1 bash -e` 
* Using gnu parallel, on multiple hosts:  
  `$ find . -name cmd | parallel --workdir $PWD --gnu --slf $PBS_NODEFILE --progress --eta  bash -e` 

### Step 5: Collect the results

`collect_results.py` will collect exit_code, RMSE and AUC for all smurff tests run and store these values in `results.csv`

### Step 6: Verify the results

`verify_results.py` will compare the `results.csv` with `expected-fail.csv` (expected failures) and `expected-pass.csv` (expected passing tests) and report.

Happy Testing!

  

