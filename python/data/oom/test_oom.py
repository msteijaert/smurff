#!/usr/bin/env python

from scipy import sparse
import numpy as np
import smurff
import matrix_io as mio
import unittest
import sys

global_verbose = False


def train_session(root, train, test, sideinfo = None):
    session = smurff.TrainSession(
                                num_latent=32,
                                burnin=1000,
                                nsamples=1000,
                                verbose = global_verbose,
                                save_freq=10,
                                save_prefix = root,
                                )
    session.addTrainAndTest(train, test, smurff.FixedNoise(1.0))
    if sideinfo is not None:
        session.addSideInfo(0, sideinfo, smurff.FixedNoise(10.), direct=True)

    predictions = session.run()
    rmse = smurff.calc_rmse(predictions)

    print("RMSE = %.2f%s" % (rmse, "" if sideinfo is None else " (with sideinfo)" ))
    return rmse

def im_prediction(predict_session, test):
    im_predictions = predict_session.predict_some(test)
    print("Macau in-matrix prediction RMSE = %.2f" % smurff.calc_rmse(im_predictions) )
    print("Predictions:")
    for p in im_predictions:
        print(p)
    print()

def oom_prediction(predict_session, sideinfo, test):
    sideinfo = sideinfo.tocsr()
    oom_predictions = [
            predict_session.predict_one((sideinfo[i, :], j), v)
            for i,j,v in zip(*sparse.find(test))
            ]
    print("Macau ouf-of-matrix prediction RMSE = %.2f" % smurff.calc_rmse(oom_predictions) )
    print("Predictions:")
    for p in oom_predictions:
        print(p)
    print()

class TestOom(unittest.TestCase):
    def test_macauoom(self):
        # ic50 = mio.read_matrix("chembl-IC50-346targets-100compounds.mm")
        # train, test = smurff.make_train_test(ic50, 0.2)
        # mio.write_matrix("chembl-IC50-346targets-100compounds-train.mm", train)
        # mio.write_matrix("chembl-IC50-346targets-100compounds-test.mm", test)
        train = mio.read_matrix("train.mm")
        train = train.tocsr()
        
        test = mio.read_matrix("test.mm")
        test = test.tocsr()

        sideinfo = mio.read_matrix("sideinfo.mm")
        sideinfo = sideinfo.tocsr()

        #train_session("bpmf_root", train, test, )
        #train_session("macau_root", train, test, sideinfo, )

        predict_session = smurff.PredictSession("save/root.ini")

        num_nonzeros_train = np.diff(train.indptr)
        test_empty = test[num_nonzeros_train == 0]
        train_empty = train[num_nonzeros_train == 0]

        print("test_empty")
        print(test_empty)
        print("train_empty")
        print(train_empty)


        im_prediction(predict_session, test_empty)
        oom_prediction(predict_session, sideinfo, test_empty)

if __name__ == "__main__":
    for arg in sys.argv:
        if (arg == "-v" or arg == "--verbose"):
            global_verbose = True
    unittest.main()
