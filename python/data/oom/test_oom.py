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
        # ic50_train, ic50_test = smurff.make_train_test(ic50, 0.2)
        # mio.write_matrix("chembl-IC50-346targets-100compounds-train.mm", ic50_train)
        # mio.write_matrix("chembl-IC50-346targets-100compounds-test.mm", ic50_test)
        ic50_train = mio.read_matrix("chembl-IC50-346targets-100compounds-train.mm")
        ic50_train = ic50_train.tocsr()
        
        ic50_test = mio.read_matrix("chembl-IC50-346targets-100compounds-test.mm")
        ic50_test = ic50_test.tocsr()

        sideinfo = mio.read_matrix("chembl-IC50-100compounds-feat.mm")
        sideinfo = sideinfo.tocsr()

        #train_session("bpmf_root", ic50_train, ic50_test, )
        #train_session("macau_root", ic50_train, ic50_test, sideinfo, )

        predict_session = smurff.PredictSession("macau_root/root.ini")

        num_nonzeros_train = np.diff(ic50_train.indptr)
        ic50_test_empty = ic50_test[num_nonzeros_train == 0]
        ic50_train_empty = ic50_train[num_nonzeros_train == 0]

        print("ic50_test_empty")
        print(ic50_test_empty)
        print("ic50_train_empty")
        print(ic50_train_empty)


        im_prediction(predict_session, ic50_test_empty)
        oom_prediction(predict_session, sideinfo, ic50_test_empty)

if __name__ == "__main__":
    for arg in sys.argv:
        if (arg == "-v" or arg == "--verbose"):
            global_verbose = True
    unittest.main()
