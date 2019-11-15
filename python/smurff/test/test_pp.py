import unittest
import numpy as np
import pandas as pd
import scipy.sparse
import smurff
import itertools
import collections

verbose = 0

# Taken from BMF_PP/postprocess_posterior_samples
def calc_posteriorMeanCovs(predict_session, axis):
    # collect U/V for all samples
    Us = [ s.latents[axis] for s in predict_session.samples() ]

    # stack them and compute mean
    Ustacked = np.stack(Us)
    mu = np.mean(Ustacked, axis = 0)

    # Compute covariance, first unstack in different way
    Uunstacked = np.squeeze(np.split(Ustacked, Ustacked.shape[2], axis = 2))
    Ucov = [ np.cov(u, rowvar = False) for u in Uunstacked ]

    # restack, shape: (K, K, N)
    Ucovstacked = np.stack(Ucov, axis = 2)

    return mu, Ucovstacked

class TestPP(unittest.TestCase):
    def test_bmf_pp(self):
        Y = scipy.sparse.rand(30, 20, 0.2)
        Y, Ytest = smurff.make_train_test(Y, 0.5)
        session = smurff.BPMFSession(Y, Ytest=Ytest, num_latent=4, verbose=verbose, burnin=5, nsamples=20, save_freq=1)
        session.run()
        predict_session = session.makePredictSession()

        for m in range(predict_session.nmodes):
            calc_mu, calc_cov = calc_posteriorMeanCovs(predict_session, m)
            sess_mu, sess_cov = predict_session.postMuCov(m)

            np.testing.assert_almost_equal(calc_mu, sess_mu)
            np.testing.assert_almost_equal(calc_cov, sess_cov)

            # print("calculated mu: ", calc_mu[0:2,0])
            # print("   session mu: ", sess_mu[0:2,0])

            # print("calculated cov ", calc_cov.shape, ": ", calc_cov[0:2,0:2,1] )
            # print("   session cov ", sess_cov.shape, ": ", sess_cov[0:2,0:2,1] )

if __name__ == '__main__':
    unittest.main()
