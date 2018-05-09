import unittest
import numpy as np
import pandas as pd
import scipy.sparse
import smurff

verbose = 1

class TestPredictSession(unittest.TestCase):
    # Python 2.7 @unittest.skip fix
    __name__ = "TestPredictSession"

    def run_train_session(self):
        #Ydense  = np.random.normal(size = (10, 20)).reshape((10,20))
        Ydense  = np.ones((10, 20))
        r       = np.random.permutation(10*20)[:40] # 40 random samples from 10*20 matrix
        Y       = scipy.sparse.coo_matrix(Ydense) # convert to sparse
        print(Y.data)
        # Y       = scipy.sparse.coo_matrix( (Y.data[r], (Y.row[r], Y.col[r])), shape=Y.shape )

        self.Ytrain, self.Ytest = smurff.make_train_test(Y, 0.5)
        self.side_info   = Ydense

        # select one test element for testing with values
        row1 = self.Ytest.row[0]
        col1 = self.Ytest.col[0]
        val1 = self.Ytest.data[0]
        self.test1 = smurff.Prediction((row1, col1), val1)

        # select one test element for testing with only side_info
        row2 = self.Ytest.row[-1]
        col2 = self.Ytest.col[-1]
        val2 = self.Ytest.data[-1]
        self.test2 = smurff.Prediction((row2, col2), val2)

        #remove that row's data
        Ycopy = self.Ytrain.tocsr()
        Ycopy.data[Ycopy.indptr[row2]:Ycopy.indptr[row2+1]] = 0.0
        Ycopy.eliminate_zeros()
        self.Ytrain = Ycopy

        print("Ytrain =", self.Ytrain)
        print("Ytest =", self.Ytest.shape)
        print("side =", self.side_info.shape)


        nmodes = len(self.Ytrain.shape)
        priors = ['normal'] * nmodes

        session = smurff.TrainSession(priors = priors, num_latent=32,
                burnin=1000, nsamples=1500, verbose=verbose,
                save_freq = 1)

        session.addTrainAndTest(self.Ytrain, self.Ytest, smurff.FixedNoise(1000))
        session.addSideInfo(0, self.side_info, smurff.FixedNoise(10000))

        session.init()
        while session.step():
            pass

        return session

    def test_simple(self):
        train_session = self.run_train_session()
        predict_session = train_session.makePredictSession()

        p1 = sorted(train_session.getTestPredictions())
        p2 = sorted(predict_session.predict_some(self.Ytest))

        one = p1[0]

        p3 = predict_session.predict_one(one.coords, one.val)
        p4 = predict_session.predict_all()

        self.assertEqual(len(p1), len(p2))

        # check train_session vs predict_session for Ytest
        self.assertEqual(p1[0].coords, p2[0].coords)
        self.assertAlmostEqual(p1[0].val, p2[0].val, places = 2)
        self.assertAlmostEqual(p1[0].pred_1sample, p2[0].pred_1sample, places = 2)
        self.assertAlmostEqual(p1[0].pred_avg, p2[0].pred_avg, places = 2)

        # check predict_session.predict_some vs predict_session.predict_one
        self.assertEqual(p1[0].coords, p3.coords)
        self.assertAlmostEqual(p1[0].val, p3.val, places = 2)
        self.assertAlmostEqual(p1[0].pred_1sample, p3.pred_1sample, places = 2)
        self.assertAlmostEqual(p1[0].pred_avg, p3.pred_avg, places = 2)

        # check predict_session.predict_some vs predict_session.predict_all
        for s in p2:
            ecoords = (Ellipsis,) + s.coords
            for p in zip(s.pred_all, p4[ecoords]):
                self.assertAlmostEqual(*p, places=2)

        last_step = predict_session.steps[-1]
        beta = last_step.betas[0]
        uhat = self.side_info.dot(beta.transpose())
        print("uhat =", uhat)
        u = last_step.latents[0].transpose()
        print("u =", u)
        umean = np.mean(last_step.latents[0], axis = 1)
        print("umean =", umean)
        print("umean+uhat =", umean + uhat)


        # check predict from side info
        two = self.test2
        print("two = ", two)
        side =  self.side_info[two.coords[0]]
        print("side = ", side)
        latent = last_step.latents[0][:,two.coords[0]]
        print("beta = ", beta)
        print("latent = ", latent)
        uhat = side.dot(beta.transpose()) + umean
        print("uhat = ", uhat)

        p5 = predict_session.predict([self.side_info[one.coords[0]], one.coords[1]])
        print("pred from side = ", p5)
        print("avg pred from side = ", np.mean(p5))
        print("var pred from side = ", np.var(p5))
        print("pred no_side = ", p3)
        for side, no_side in zip(p5, p3.pred_all):
            self.assertAlmostEqual(side, no_side, places=2)

        p1_rmse_avg = smurff.calc_rmse(p1)
        p2_rmse_avg = smurff.calc_rmse(p2)

        self.assertAlmostEqual(train_session.getRmseAvg(), p2_rmse_avg, places = 2)
        self.assertAlmostEqual(train_session.getRmseAvg(), p1_rmse_avg, places = 2)

if __name__ == '__main__':
    unittest.main()
