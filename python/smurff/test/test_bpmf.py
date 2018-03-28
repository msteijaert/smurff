import unittest
import smurff
import scipy.sparse
import numpy as np

class TestBPMF(unittest.TestCase):
    def test_bpmf(self):
        X = scipy.sparse.rand(15, 10, 0.2)
        smurff.smurff(X,
                      priors=['normal', 'normal'],
                      side_info=[None, None],
                      aux_data=[[], []],
                      num_latent = 10,
                      burnin=10,
                      nsamples=15,
                      verbose=False)

    def test_bpmf2(self):
        X = scipy.sparse.rand(15, 10, 0.2)
        Xt = scipy.sparse.rand(15, 10, 0.1)
        smurff.smurff(X,
                      Ytest=Xt,
                      priors=['normal', 'normal'],
                      side_info=[None, None],
                      aux_data=[[], []],
                      num_latent=10,
                      burnin=10,
                      nsamples=15,
                      verbose=False)

    def test_bpmf_numerictest(self):
        X = scipy.sparse.rand(15, 10, 0.2)
        Xt = 0.3
        X, Xt = smurff.make_train_test(X, Xt)
        smurff.smurff(X,
                      Ytest=Xt,
                      priors=['normal', 'normal'],
                      side_info=[None, None],
                      aux_data=[[], []],
                      num_latent=10,
                      burnin=10,
                      nsamples=15,
                      verbose=False)

    def test_macau_sparse(self):
         X = scipy.sparse.rand(15, 10, 0.2)
         Xt = scipy.sparse.rand(15, 10, 0.1)
         F = scipy.sparse.rand(15, 2, 0.5)
         smurff.smurff(X,
                       Ytest=Xt,
                       priors=['macau', 'normal'],
                       side_info=[[(F, None, None)], None],
                       aux_data=[None, None],
                       num_latent=5,
                       burnin=10,
                       nsamples=5,
                       verbose=False)

    def test_macau_binsparse(self):
        X = scipy.sparse.rand(15, 10, 0.2)
        Xt = scipy.sparse.rand(15, 10, 0.1)
        F = scipy.sparse.rand(15, 2, 0.5)
        F.data[:] = 1
        smurff.smurff(X,
                      Ytest=Xt,
                      priors=['macau', 'normal'],
                      side_info=[[(F, None, None)], None],
                      aux_data=[None, None],
                      num_latent=5,
                      burnin=10,
                      nsamples=5,
                      verbose=False)

    def test_make_train_test(self):
        X = scipy.sparse.rand(15, 10, 0.2)
        Xtr, Xte = smurff.make_train_test(X, 0.5)
        self.assertEqual(X.nnz, Xtr.nnz + Xte.nnz)
        diff = np.linalg.norm( (X - Xtr - Xte).todense() )
        self.assertEqual(diff, 0.0)

if __name__ == '__main__':
    unittest.main()
