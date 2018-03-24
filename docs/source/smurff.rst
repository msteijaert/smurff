
SMURFF Tutorial
===============

In these examples we use ChEMBL dataset for compound-proteins activities
(IC50). The IC50 values and ECFP fingerprints can be downloaded from
these two urls:

.. code:: bash

    %%bash
    wget http://homes.esat.kuleuven.be/~jsimm/chembl-IC50-346targets.mm
    wget http://homes.esat.kuleuven.be/~jsimm/chembl-IC50-compound-feat.mm

Matrix Factorization Model
--------------------------

The matrix factorization models cell **``Y[i,j]``** by the inner product
of the latents

.. math::  Y_{ij} ∼ N(\textbf{u}_{i} \textbf{v}_{j} + mean, \alpha^{-1}) 

where :math:`\textbf{u}_{i}` and :math:`\textbf{v}_{j}` are the latent
vector for i-th row and j-th column, and :math:`\alpha` is the precision
of the observation noise. The model also uses a fixed global mean for
the whole matrix.

Matrix Factorization with Side Information
------------------------------------------

In this example we use MCMC (Gibbs) sampling to perform factorization of
the compound x protein IC50 matrix by using ECFP features as side
information on the compounds.

.. code:: ipython3

    import smurff
    import scipy.io
    
    import scipy.sparse
    import numpy
    
    ## loading data
    ic50 = scipy.io.mmread("chembl-IC50-346targets.mm")
    ecfp = scipy.io.mmread("chembl-IC50-compound-feat.mm")
    
    ## creating train and test sets
    ic50_train, ic50_test = smurff.make_train_test(ic50, 0.2)

.. code:: ipython3

    ## running factorization (Macau)
    result = smurff.smurff(Y          = ic50_train,
                           Ytest      = ic50_test,
                           priors     = ['macau', 'normal'],
                           side_info  = [ecfp, None],
                           aux_data   = [[], []],
                           verbose    = 2,
                           num_latent = 2,
                           precision  = 5.0,
                           burnin     = 40,
                           save_freq  = 10,
                           nsamples   = 100)
    print(result.rmse)

Input matrix for **``Y``** is a sparse scipy matrix (either coo\_matrix,
csr\_matrix or csc\_matrix).

In this example, we have assigned 20% of the IC50 data to the test set
by setting **``Ytest = 0.2``**. If you want to use a predefined test
data, set **``Ytest = my_test_matrix``**, where the matrix is a sparse
matrix of the same size as **``Y``**. Here we have used burn-in of 400
samples for the Gibbs sampler and then collected 1600 samples from the
model. This is usually sufficient. For quick runs smaller numbers can be
used, like **``burnin = 100, nsamples = 500``**.

The parameter **``side_info = [ecfp, None]``** sets the side information
for rows and columns, respectively. In this example we only use side
information for the compounds (rows of the matrix).

The **``precision = 5.0``** specifies the precision of the IC50
observations, i.e., 1 / variance.

When the run has completed you can check the **``result``** object and
its **``predictions``** field, which is a list of **``ResultItem``**.

.. code:: ipython3

    print("RMSE: {0}".format(result.rmse))
    print(result.predictions[0])

Univariate sampler
------------------

SMURFF also includes an option to use a **very fast** univariate
sampler, i.e., instead of sampling blocks of variables jointly it
samples each individually. An example:

.. code:: ipython3

    result = smurff.smurff(Y          = ic50_train,
                           Ytest      = ic50_test,
                           priors     = ['macauone', 'normal'],
                           side_info  = [ecfp, None],
                           aux_data   = [[], []],
                           num_latent = 32,
                           precision  = 5.0,
                           burnin     = 500,
                           nsamples   = 3500)

When using it we recommend using larger values for **``burnin``** and
**``nsamples``**, because the univariate sampler mixes slower than the
blocked sampler.

Adaptive noise
--------------

In the previous examples we fixed the observation noise by specifying
**``precision = 5.0``**. Instead we can also allow the model to
automatically determine the precision of the noise by setting
signal-to-noise ratio parameters **``sn_init``** and **``sn_max``**.

**``sn_init``** is an initial signal-to-noise ratio.

**``sn_max``** is the maximum allowed signal-to-noise ratio. This means
that if the updated precision would imply a higher signal-to-noise ratio
than **``sn_max``**, then the precision value is set to
**``(sn_max + 1.0) / Yvar``** where **``Yvar``** is the variance of the
training dataset **``Y``**.

.. code:: ipython3

    result = smurff.smurff(Y          = ic50_train,
                           Ytest      = ic50_test,
                           priors     = ['macauone', 'normal'],
                           side_info  = [ecfp, None],
                           aux_data   = [[], []],
                           num_latent = 32,
                           sn_init    = 0,
                           sn_max     = 1,
                           burnin     = 500,
                           nsamples   = 3500)

Binary matrices
---------------

SMURFF can also factorize binary matrices (with or without side
information). As an input the sparse matrix should only contain values
of 0 or 1. To factorize them we employ probit noise model that can be
enabled by setting **``threshold``** parameter.

Care has to be taken to make input to the model, as operating with
sparse matrices can drop real 0 measurements. In the below example, we
first copy the matrix (line 9) and then threshold the data to binary
(line 10).

.. code:: ipython3

    ## using activity threshold pIC50 > 6.5
    act = ic50
    act.data = act.data > 6.5
    act_train, act_test = smurff.make_train_test(act, 0.5)
    
    ## running factorization (Macau)
    result = smurff.smurff(Y          = act_train,
                           Ytest      = act_test,
                           priors     = ['macau', 'normal'],
                           side_info  = [ecfp, None],
                           aux_data   = [[], []],
                           num_latent = 32,
                           threshold  = 0.5,
                           burnin     = 500,
                           nsamples   = 3500)

Matrix Factorization without Side Information
---------------------------------------------

You can run SMURFF without side information. But you should use Bayesian
Matrix Factorization (BPMF) instead of macau prior.

So you should set all **``side_info``** values to **``None``** and
update **``priors``** parameter to have only **``'normal'``** values.

.. code:: ipython3

    result = smurff.smurff(Y          = ic50_train,
                           Ytest      = ic50_test,
                           priors     = ['normal', 'normal'],
                           side_info  = [None, None],
                           aux_data   = [[], []],
                           num_latent = 32,
                           precision  = 5.0,
                           burnin     = 200,
                           nsamples   = 800)

Tensor Factorization
--------------------

SMURFF also supports tensor factorization with and without side
information on any of the modes. Tensor can be thought as generalization
of matrix to relations with more than two items. For example 3-tensor of
**``drug x cell x gene``** could express the effect of a drug on the
given cell and gene. In this case the prediction for the element
**``Yhat[i,j,k]``**\ \* is given by

.. math::  \hat{Y}_{ijk} = \sum_{d=1}^{D}u^{(1)}_{d,i}u^{(2)}_{d,j}u^{(3)}_{d,k} + mean 

Visually the model can be represented as follows:

.. raw:: html

   <center>

Tensor model predicts Yhat[i,j,k] by multiplying all latent vectors
together element-wise and then taking the sum along the latent dimension
(figure omits the global mean).

.. raw:: html

   </center>

For tensors SMURFF packages uses Pandas **``DataFrame``** where each row
stores the coordinate and the value of a known cell in the tensor.
Specifically, the integer columns in the DataFrame give the coordinate
of the cell and **``float``** (or double) column stores the value in the
cell (the order of the columns does not matter). The coordinates are
0-based.

Here is a simple toy example with factorizing a 3-tensor with side
information on the first mode.

.. code:: ipython3

    import numpy as np
    import pandas as pd
    import scipy.sparse
    import smurff
    import itertools
    
    ## generating toy data
    A = np.random.randn(15, 2)
    B = np.random.randn(3, 2)
    C = np.random.randn(2, 2)
    
    idx = list( itertools.product(np.arange(A.shape[0]),
                                  np.arange(B.shape[0]),
                                  np.arange(C.shape[0])) )
    df  = pd.DataFrame( np.asarray(idx), columns=["A", "B", "C"])
    df["value"] = np.array([ np.sum(A[i[0], :] * B[i[1], :] * C[i[2], :]) for i in idx ])
    
    ## assigning 20% of the cells to test set
    Ytrain, Ytest = smurff.make_train_test_df(df, 0.2)
    
    ## for artificial dataset using small values for burnin, nsamples and num_latents is fine
    results = smurff.smurff(Ytrain,
                            Ytest=Ytest,
                            priors=['normal', 'normal', 'normal'],
                            side_info=[None, None, None],
                            aux_data=[[], [], []],
                            num_latent=4,
                            precision=50,
                            burnin=20,
                            nsamples=20)
