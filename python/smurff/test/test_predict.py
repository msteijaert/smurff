import unittest
import numpy as np
import pandas as pd
import scipy.sparse
import smurff

verbose = 0

class TestNoiseModels(unittest.TestCase):
    # Python 2.7 @unittest.skip fix
    __name__ = "TestNoiseModels"

    def run_train_session(self):
        Y = scipy.sparse.rand(15, 10, 0.2)
        self.Ytrain, self.Ytest = smurff.make_train_test(Y, 0.5)
        nmodes = len(self.Ytrain.shape)
        priors = ['normal'] * nmodes

        session = smurff.TrainSession(priors = priors, num_latent=4,
                burnin=10, nsamples=15, verbose=verbose,
                save_freq = 1)

        session.addTrainAndTest(self.Ytrain, self.Ytest)

        session.init()
        while session.step():
            pass

        return session

    # 5 different noise configs

    def test_simple(self):
        train_session = self.run_train_session()
        predict_session = train_session.makePredictSession()

        p1 = train_session.getTestPredictions()
        p2 = predict_session.predict_some(self.Ytest)

        p1 = sorted(p1)
        p2 = sorted(p2)

        p3 = predict_session.predict_one(p1[0].coords, p1[0].val)
        p4 = predict_session.predict_all()

        self.assertEqual(len(p1), len(p2))

        self.assertEqual(p1[0].coords, p2[0].coords)
        self.assertAlmostEqual(p1[0].val, p2[0].val, places = 2)
        self.assertAlmostEqual(p1[0].pred_1sample, p2[0].pred_1sample, places = 2)
        self.assertAlmostEqual(p1[0].pred_avg, p2[0].pred_avg, places = 2)

        self.assertEqual(p1[0].coords, p3.coords)
        self.assertAlmostEqual(p1[0].val, p3.val, places = 2)
        self.assertAlmostEqual(p1[0].pred_1sample, p3.pred_1sample, places = 2)
        self.assertAlmostEqual(p1[0].pred_avg, p3.pred_avg, places = 2)

        ecoords = (Ellipsis,) + p2[0].coords
        [ self.assertAlmostEqual(*p, places=2) for p in zip(p2[0].pred_all, p4[ecoords]) ]

        p1_rmse_avg = smurff.calc_rmse(p1)
        p2_rmse_avg = smurff.calc_rmse(p2)

        self.assertAlmostEqual(train_session.getRmseAvg(), p2_rmse_avg, places = 2)
        self.assertAlmostEqual(train_session.getRmseAvg(), p1_rmse_avg, places = 2)


if __name__ == '__main__':
    unittest.main()
