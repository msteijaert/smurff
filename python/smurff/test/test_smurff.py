import unittest
import numpy as np
import pandas as pd
import scipy.sparse
import smurff
import itertools
import collections

class TestSmurff(unittest.TestCase):

    # Python 2.7 @unittest.skip fix
    __name__ = "TestSmurff"

    def test_bpmf(self):
        Y = scipy.sparse.rand(10, 20, 0.2)
        Y, Ytest = smurff.make_train_test(Y, 0.5)
        results = smurff.smurff(Y,
                                Ynoise=('fixed', 1.0, None, None, None),
                                Ytest=Ytest,
                                priors=['normal', 'normal'],
                                num_latent=4,
                                verbose=False,
                                burnin=50,
                                nsamples=50)
        self.assertEqual(Ytest.nnz, len(results.predictions))

    def test_bpmf_numerictest(self):
        X = scipy.sparse.rand(15, 10, 0.2)
        Xt = 0.3
        X, Xt = smurff.make_train_test(X, Xt)
        smurff.smurff(X,
                      Ytest=Xt,
                      priors=['normal', 'normal'],
                      num_latent=10,
                      burnin=10,
                      nsamples=15,
                      verbose=False)

    def test_macau(self):
        Ydense  = np.random.rand(10, 20)
        r       = np.random.permutation(10*20)[:40] # 40 random samples from 10*20 matrix
        side1   = Ydense[:,1:2]
        side2   = Ydense[1:2,:].transpose()
        Y       = scipy.sparse.coo_matrix(Ydense) # convert to sparse
        Y       = scipy.sparse.coo_matrix( (Y.data[r], (Y.row[r], Y.col[r])), shape=Y.shape )
        Y, Ytest = smurff.make_train_test(Y, 0.5)

        results = smurff.smurff(Y,
                                Ytest=Ytest,
                                priors=['macau', 'macau'],
                                side_info=[[(side1, None, None)], [(side2, None, None)]],
                                side_info_noises=[[('fixed', 1.0, None, None, None)], [('adaptive', None, 0.5, 1.0, None)]],
                                num_latent=4,
                                verbose=False,
                                burnin=50,
                                nsamples=50)
        #self.assertEqual(Ytest.nnz, len(results.predictions))

    def test_macau_side_bin(self):
        X = scipy.sparse.rand(15, 10, 0.2)
        Xt = scipy.sparse.rand(15, 10, 0.1)
        F = scipy.sparse.rand(15, 2, 0.5)
        F.data[:] = 1
        smurff.smurff(X,
                      Ytest=Xt,
                      priors=['macau', 'normal'],
                      side_info=[[(F, None, None)], None],
                      aux_data=[None, []],
                      num_latent=5,
                      burnin=10,
                      nsamples=5,
                      verbose=False)

    def test_macau_dense(self):
        Y  = scipy.sparse.rand(15, 10, 0.2)
        Yt = scipy.sparse.rand(15, 10, 0.1)
        F  = np.random.randn(15, 2)
        smurff.smurff(Y,
                      Ytest=Yt,
                      priors=['macau', 'normal'],
                      side_info=[[(F, None, True)], None],
                      num_latent=5,
                      burnin=10,
                      nsamples=5,
                      verbose=False)

    @unittest.skip
    def test_macau_dense_probit(self):
        A = np.random.randn(25, 2)
        B = np.random.randn(3, 2)

        idx = list( itertools.product(np.arange(A.shape[0]), np.arange(B.shape[0])) )
        df  = pd.DataFrame( np.asarray(idx), columns=["A", "B"])
        df["value"] = (np.array([ np.sum(A[i[0], :] * B[i[1], :]) for i in idx ]) > 0.0).astype(np.float64)
        Ytrain, Ytest = smurff.make_train_test_df(df, 0.2)

        results = smurff.smurff(Y=Ytrain,
                                Ynoise=('probit', None, None, None, 0.5),
                                Ytest=Ytest,
                                priors=['macau', 'normal'],
                                prior_noises=[('probit', None, None, None, 0.5), ('fixed', 1.0, None, None, None)],
                                side_info=[[(A, None, None)], None],
                                num_latent=4,
                                burnin=20,
                                nsamples=20,
                                verbose=False)

        self.assertTrue(results.rmse > 0.55,
                        msg="Probit factorization (with dense side) gave AUC below 0.55 (%f)." % results.rmse)

    def test_macau_univariate(self):
        Y = scipy.sparse.rand(10, 20, 0.2)
        Y, Ytest = smurff.make_train_test(Y, 0.5)
        side1   = scipy.sparse.coo_matrix( np.random.rand(10, 2) )
        side2   = scipy.sparse.coo_matrix( np.random.rand(20, 3) )

        results = smurff.smurff(Y,
                                Ytest=Ytest,
                                priors=['macauone', 'macauone'],
                                side_info=[[(side1, None, None)], [(side2, None, None)]],
                                num_latent=4,
                                verbose=False,
                                burnin=50,
                                nsamples=50)
        self.assertEqual(Ytest.nnz, len(results.predictions))

    def test_too_many_sides(self):
        Y = scipy.sparse.rand(10, 20, 0.2)
        with self.assertRaises(RuntimeError):
            smurff.smurff(Y,
                          priors=['normal', 'normal', 'normal'],
                          side_info=[None, None, None],
                          aux_data=[[], [], []],
                          verbose = False)

    def test_bpmf_emptytest(self):
        X = scipy.sparse.rand(15, 10, 0.2)
        smurff.smurff(X,
                      priors=['normal', 'normal'],
                      num_latent=10,
                      burnin=10,
                      nsamples=15,
                      verbose=False)

    def test_bpmf_emptytest_probit(self):
        X = scipy.sparse.rand(15, 10, 0.2)
        X.data = X.data > 0.5
        smurff.smurff(X,
                      Ynoise=('probit', None, None, None, 0.5),
                      priors=['normal', 'normal'],
                      num_latent=10,
                      burnin=10,
                      nsamples=15,
                      verbose=False)

    def test_make_train_test(self):
        X = scipy.sparse.rand(15, 10, 0.2)
        Xtr, Xte = smurff.make_train_test(X, 0.5)
        self.assertEqual(X.nnz, Xtr.nnz + Xte.nnz)
        diff = np.linalg.norm( (X - Xtr - Xte).todense() )
        self.assertEqual(diff, 0.0)

    def test_make_train_test_df(self):
        idx = list( itertools.product(np.arange(10), np.arange(8), np.arange(3) ))
        df  = pd.DataFrame( np.asarray(idx), columns=["A", "B", "C"])
        df["value"] = np.arange(10.0 * 8.0 * 3.0)

        Ytr, Yte = smurff.make_train_test_df(df, 0.4)
        self.assertEqual(Ytr.shape[0], df.shape[0] * 0.6)
        self.assertEqual(Yte.shape[0], df.shape[0] * 0.4)

        A1 = np.zeros( (10, 8, 3) )
        A2 = np.zeros( (10, 8, 3) )
        A1[df.A, df.B, df.C] = df.value
        A2[Ytr.A, Ytr.B, Ytr.C] = Ytr.value
        A2[Yte.A, Yte.B, Yte.C] = Yte.value

        self.assertTrue(np.allclose(A1, A2))

    def test_bpmf_tensor(self):
        np.random.seed(1234)
        Y = pd.DataFrame({
            "A": np.random.randint(0, 5, 7),
            "B": np.random.randint(0, 4, 7),
            "C": np.random.randint(0, 3, 7),
            "value": np.random.randn(7)
        })
        Ytest = pd.DataFrame({
            "A": np.random.randint(0, 5, 5),
            "B": np.random.randint(0, 4, 5),
            "C": np.random.randint(0, 3, 5),
            "value": np.random.randn(5)
        })

        results = smurff.smurff(Y,
                                Ytest=Ytest,
                                priors=['normal', 'normal', 'normal'],
                                num_latent=4,
                                verbose=False,
                                burnin=50,
                                nsamples=50)

    def test_bpmf_sparse_matrix_sparse_2d_tensor(self):
        np.random.seed(1234)

        # Generate train matrix rows, cols and vals
        train_shape = (5, 5)
        train_rows = np.random.randint(0, 5, 7)
        train_cols = np.random.randint(0, 4, 7)
        train_vals = np.random.randn(7)

        # Generate test matrix rows, cols and vals
        test_shape = (5, 5)
        test_rows = np.random.randint(0, 5, 5)
        test_cols = np.random.randint(0, 4, 5)
        test_vals = np.random.randn(5)

        # Create train and test sparse matrices
        train_sparse_matrix = scipy.sparse.coo_matrix((train_vals, (train_rows, train_cols)), train_shape)
        test_sparse_matrix = scipy.sparse.coo_matrix((test_vals, (test_rows, test_cols)), test_shape)

        # Force NNZ recalculation to remove duplicate coordinates because of random generation
        train_sparse_matrix.count_nonzero()
        test_sparse_matrix.count_nonzero()

        # Create train and test sparse tensors
        train_sparse_tensor = pd.DataFrame({
            '0': train_sparse_matrix.row,
            '1': train_sparse_matrix.col,
            'v': train_sparse_matrix.data
        })
        test_sparse_tensor = pd.DataFrame({
            '0': test_sparse_matrix.row,
            '1': test_sparse_matrix.col,
            'v': test_sparse_matrix.data
        })

        # Run SMURFF
        sparse_matrix_results = smurff.smurff(train_sparse_matrix,
                                              Ytest=test_sparse_matrix,
                                              data_shape=train_shape,
                                              priors=['normal', 'normal'],
                                              num_latent=4,
                                              verbose=False,
                                              burnin=50,
                                              nsamples=50,
                                              seed=1234)

        sparse_tensor_results = smurff.smurff(train_sparse_tensor,
                                              Ytest=test_sparse_tensor,
                                              data_shape=train_shape,
                                              priors=['normal', 'normal'],
                                              num_latent=4,
                                              verbose=False,
                                              burnin=50,
                                              nsamples=50,
                                              seed=1234)

        # Transfrom SMURFF results to dictionary of coords and predicted values
        sparse_matrix_results_dict = collections.OrderedDict((p.coords, p.pred_1sample) for p in sparse_matrix_results.predictions)
        sparse_tensor_results_dict = collections.OrderedDict((p.coords, p.pred_1sample) for p in sparse_tensor_results.predictions)

        self.assertEqual(len(sparse_matrix_results_dict), len(sparse_tensor_results_dict))
        self.assertEqual(sparse_tensor_results_dict.keys(), sparse_tensor_results_dict.keys())
        for coords, matrix_pred_1sample in sparse_matrix_results_dict.items():
            tensor_pred_1sample = sparse_tensor_results_dict[coords]
            self.assertAlmostEqual(matrix_pred_1sample, tensor_pred_1sample)

    def test_bpmf_dense_matrix_sparse_2d_tensor(self):
        np.random.seed(1234)

        # Generate train dense matrix
        train_shape = (5 ,5)
        train_sparse_matrix = scipy.sparse.random(5, 5, density=1.0)
        train_dense_matrix = train_sparse_matrix.todense()

        # Generate test sparse matrix
        test_shape = (5, 5)
        test_rows = np.random.randint(0, 5, 5)
        test_cols = np.random.randint(0, 4, 5)
        test_vals = np.random.randn(5)
        test_sparse_matrix = scipy.sparse.coo_matrix((test_vals, (test_rows, test_cols)), test_shape)

        # Create train and test sparse tensors
        train_sparse_tensor = pd.DataFrame({
            '0': train_sparse_matrix.row,
            '1': train_sparse_matrix.col,
            'v': train_sparse_matrix.data
        })
        test_sparse_tensor = pd.DataFrame({
            '0': test_sparse_matrix.row,
            '1': test_sparse_matrix.col,
            'v': test_sparse_matrix.data
        })

        # Run SMURFF
        sparse_matrix_results = smurff.smurff(train_dense_matrix,
                                              Ytest=test_sparse_matrix,
                                              priors=['normal', 'normal'],
                                              data_shape=train_shape,
                                              num_latent=4,
                                              verbose=False,
                                              burnin=50,
                                              nsamples=50,
                                              seed=1234)

        sparse_tensor_results = smurff.smurff(train_sparse_tensor,
                                              Ytest=test_sparse_tensor,
                                              data_shape=train_shape,
                                              priors=['normal', 'normal'],
                                              num_latent=4,
                                              verbose=False,
                                              burnin=50,
                                              nsamples=50,
                                              seed=1234)

        # Transfrom SMURFF results to dictionary of coords and predicted values
        sparse_matrix_results_dict = collections.OrderedDict((p.coords, p.pred_1sample) for p in sparse_matrix_results.predictions)
        sparse_tensor_results_dict = collections.OrderedDict((p.coords, p.pred_1sample) for p in sparse_tensor_results.predictions)

        self.assertEqual(len(sparse_matrix_results_dict), len(sparse_tensor_results_dict))
        self.assertEqual(sparse_tensor_results_dict.keys(), sparse_tensor_results_dict.keys())
        for coords, matrix_pred_1sample in sparse_matrix_results_dict.items():
            tensor_pred_1sample = sparse_tensor_results_dict[coords]
            self.assertAlmostEqual(matrix_pred_1sample, tensor_pred_1sample)

    def test_bpmf_tensor2(self):
        A = np.random.randn(15, 2)
        B = np.random.randn(20, 2)
        C = np.random.randn(3, 2)

        idx = list( itertools.product(np.arange(A.shape[0]), np.arange(B.shape[0]), np.arange(C.shape[0])) )
        df  = pd.DataFrame( np.asarray(idx), columns=["A", "B", "C"])
        df["value"] = np.array([ np.sum(A[i[0], :] * B[i[1], :] * C[i[2], :]) for i in idx ])
        Ytrain, Ytest = smurff.make_train_test_df(df, 0.2)

        results = smurff.smurff(Y=Ytrain,
                                Ynoise=('fixed', 50, None, None, None),
                                Ytest=Ytest,
                                priors=['normal', 'normal', 'normal'],
                                num_latent=4,
                                verbose=False,
                                burnin=20,
                                nsamples=20)

        self.assertTrue(results.rmse < 0.5,
                        msg="Tensor factorization gave RMSE above 0.5 (%f)." % results.rmse)

    def test_bpmf_tensor3(self):
        A = np.random.randn(15, 2)
        B = np.random.randn(20, 2)
        C = np.random.randn(1, 2)

        idx = list( itertools.product(np.arange(A.shape[0]), np.arange(B.shape[0]), np.arange(C.shape[0])) )
        df  = pd.DataFrame( np.asarray(idx), columns=["A", "B", "C"])
        df["value"] = np.array([ np.sum(A[i[0], :] * B[i[1], :] * C[i[2], :]) for i in idx ])
        Ytrain, Ytest = smurff.make_train_test_df(df, 0.2)

        results = smurff.smurff(Y=Ytrain,
                                Ytest=Ytest,
                                priors=['normal', 'normal', 'normal'],
                                num_latent=4,
                                verbose=False,
                                burnin=20,
                                nsamples=20)

        self.assertTrue(results.rmse < 0.5,
                        msg="Tensor factorization gave RMSE above 0.5 (%f)." % results.rmse)

        Ytrain_sp = scipy.sparse.coo_matrix( (Ytrain.value, (Ytrain.A, Ytrain.B) ) )
        Ytest_sp  = scipy.sparse.coo_matrix( (Ytest.value,  (Ytest.A, Ytest.B) ) )

        results_mat = smurff.smurff(Y=Ytrain_sp,
                                    Ytest=Ytest_sp,
                                    priors=['normal', 'normal'],
                                    num_latent=4,
                                    verbose=False,
                                    burnin=20,
                                    nsamples=20)

    @unittest.skip
    def test_macau_tensor(self):
        A = np.random.randn(15, 2)
        B = np.random.randn(3, 2)
        C = np.random.randn(2, 2)

        idx = list( itertools.product(np.arange(A.shape[0]), np.arange(B.shape[0]), np.arange(C.shape[0])) )
        df  = pd.DataFrame( np.asarray(idx), columns=["A", "B", "C"])
        df["value"] = np.array([ np.sum(A[i[0], :] * B[i[1], :] * C[i[2], :]) for i in idx ])
        Ytrain, Ytest = smurff.make_train_test_df(df, 0.2)

        Acoo = scipy.sparse.coo_matrix(A)

        results = smurff.smurff(Y = Ytrain, Ytest = Ytest, side=[('macau', [Acoo]), ('normal', []), ('normal', [])],
                                num_latent = 4, verbose = False, burnin = 20, nsamples = 20)

        self.assertTrue(results.rmse < 0.5,
                        msg="Tensor factorization gave RMSE above 0.5 (%f)." % results.rmse)

    @unittest.skip
    def test_macau_tensor_univariate(self):
        A = np.random.randn(30, 2)
        B = np.random.randn(4, 2)
        C = np.random.randn(2, 2)

        idx = list( itertools.product(np.arange(A.shape[0]), np.arange(B.shape[0]), np.arange(C.shape[0])) )
        df  = pd.DataFrame( np.asarray(idx), columns=["A", "B", "C"])
        df["value"] = np.array([ np.sum(A[i[0], :] * B[i[1], :] * C[i[2], :]) for i in idx ])
        Ytrain, Ytest = smurff.make_train_test_df(df, 0.2)

        Acoo = scipy.sparse.coo_matrix(A)

        results = smurff.smurff(Y=Ytrain,
                                Ytest=Ytest,
                                priors=['macauone', 'normal', 'normal'],
                                side_info=[Acoo, None, None],
                                aux_data=[None, [], []],
                                num_latent=4,
                                verbose=False,
                                burnin=20,
                                nsamples=20)

        self.assertTrue(results.rmse < 0.5,
                        msg="Tensor factorization gave RMSE above 0.5 (%f)." % results.rmse)

    def test_macau_tensor_empty(self):
        A = np.random.randn(30, 2)
        B = np.random.randn(4, 2)
        C = np.random.randn(2, 2)

        idx = list( itertools.product(np.arange(A.shape[0]), np.arange(B.shape[0]), np.arange(C.shape[0])) )
        df  = pd.DataFrame( np.asarray(idx), columns=["A", "B", "C"])
        df["value"] = np.array([ np.sum(A[i[0], :] * B[i[1], :] * C[i[2], :]) for i in idx ])

        Acoo = scipy.sparse.coo_matrix(A)
        r0 = smurff.smurff(df,
                           priors=['normal', 'normal', 'normal'],
                           num_latent=2,
                           burnin=5,
                           nsamples=5,
                           verbose=False)

        self.assertTrue( np.isnan(r0.rmse) )


if __name__ == '__main__':
    unittest.main()
