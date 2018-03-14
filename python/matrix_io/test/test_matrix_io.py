import os
import shutil
import unittest
import matrix_io

import numpy
import scipy
import scipy.io

class TestMatrixIO(unittest.TestCase):
    TEMP_DIR_NAME = 'tmp'

    @classmethod
    def setUpClass(cls):
        shutil.rmtree(cls.TEMP_DIR_NAME, ignore_errors=True)
        os.makedirs(cls.TEMP_DIR_NAME)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.TEMP_DIR_NAME, ignore_errors=True)

    def test_dense_float64(self):
        matrix_filename = "test_dense_float64.ddm"
        matrix_relative_path = "{}/{}".format(self.TEMP_DIR_NAME, matrix_filename)
        expected_matrix = numpy.random.randn(10, 20)
        matrix_io.write_dense_float64(matrix_relative_path, expected_matrix)
        actual_matrix = matrix_io.read_dense_float64(matrix_relative_path)
        self.assertTrue(numpy.array_equal(actual_matrix, expected_matrix))

    def test_sparse_float64(self):
        matrix_filename = "test_sparse_float64.sdm"
        matrix_relative_path = "{}/{}".format(self.TEMP_DIR_NAME, matrix_filename)
        expected_matrix = scipy.sparse.rand(10, 20, 0.5)
        matrix_io.write_sparse_float64(matrix_relative_path, expected_matrix)
        actual_matrix = matrix_io.read_sparse_float64(matrix_relative_path)
        self.assertTrue((expected_matrix != actual_matrix).nnz == 0)

    def test_sparse_binary_matrix(self):
        matrix_filename = "test_sparse_binary_matrix.sbm"
        matrix_relative_path = "{}/{}".format(self.TEMP_DIR_NAME, matrix_filename)
        expected_dense_matrix = numpy.random.randint(0, 2, size=(10, 20))
        expected_sparse_matrix = scipy.sparse.coo_matrix(expected_dense_matrix)
        matrix_io.write_sparse_binary_matrix(matrix_relative_path, expected_sparse_matrix)
        actual_matrix = matrix_io.read_sparse_binary_matrix(matrix_relative_path)
        self.assertTrue((expected_sparse_matrix != actual_matrix).nnz == 0)

    def test_dense_float64_matrix_as_tensor(self):
        matrix_filename = "test_dense_float64_matrix_as_tensor.ddt"
        matrix_relative_path = "{}/{}".format(self.TEMP_DIR_NAME, matrix_filename)
        expected_matrix = numpy.random.randn(10, 20)
        matrix_io.write_dense_float64_matrix_as_tensor(matrix_relative_path, expected_matrix)
        actual_matrix = matrix_io.read_dense_float64_tensor_as_matrix(matrix_relative_path)
        self.assertTrue(numpy.array_equal(actual_matrix, expected_matrix))

    def test_sparse_float64_matrix_as_tensor(self):
        matrix_filename = "test_sparse_float64_matrix_as_tensor.sdt"
        matrix_relative_path = "{}/{}".format(self.TEMP_DIR_NAME, matrix_filename)
        expected_matrix = scipy.sparse.rand(10, 20, 0.5)
        matrix_io.write_sparse_float64_matrix_as_tensor(matrix_relative_path, expected_matrix)
        actual_matrix = matrix_io.read_sparse_float64_tensor_as_matrix(matrix_relative_path)
        self.assertTrue((expected_matrix != actual_matrix).nnz == 0)

    def test_csv(self):
        matrix_filename = "test_csv.csv"
        matrix_relative_path = "{}/{}".format(self.TEMP_DIR_NAME, matrix_filename)
        expected_matrix = numpy.random.randn(10, 20)
        matrix_io.write_csv(matrix_relative_path, expected_matrix)
        actual_matrix = matrix_io.read_csv(matrix_relative_path)
        self.assertTrue(numpy.allclose(actual_matrix, expected_matrix))

    def test_dense_mtx(self):
        matrix_filename = "test_dense_mtx.mtx"
        matrix_relative_path = "{}/{}".format(self.TEMP_DIR_NAME, matrix_filename)
        expected_matrix = numpy.random.randn(10, 20)
        matrix_io.my_mmwrite(matrix_relative_path, expected_matrix)
        actual_matrix = scipy.io.mmread(matrix_relative_path)
        self.assertTrue(numpy.allclose(actual_matrix, expected_matrix))

    def test_sparse_mtx(self):
        matrix_filename = "test_sparse_mtx.mtx"
        matrix_relative_path = "{}/{}".format(self.TEMP_DIR_NAME, matrix_filename)
        expected_matrix = scipy.sparse.rand(10, 20, 0.5)
        matrix_io.my_mmwrite(matrix_relative_path, expected_matrix)
        actual_matrix = scipy.io.mmread(matrix_relative_path)
        self.assertTrue(numpy.allclose(actual_matrix.todense(), expected_matrix.todense()))

    def test_matrix_ddm(self):
        matrix_filename = "test_matrix_ddt.ddm"
        matrix_relative_path = "{}/{}".format(self.TEMP_DIR_NAME, matrix_filename)
        expected_matrix = numpy.random.randn(10, 20)
        matrix_io.write_matrix(matrix_relative_path, expected_matrix)
        actual_matrix = matrix_io.read_matrix(matrix_relative_path)
        self.assertTrue(numpy.array_equal(actual_matrix, expected_matrix))

    def test_matrix_sdm(self):
        matrix_filename = "test_matrix_sdm.sdm"
        matrix_relative_path = "{}/{}".format(self.TEMP_DIR_NAME, matrix_filename)
        expected_matrix = scipy.sparse.rand(10, 20, 0.5)
        matrix_io.write_matrix(matrix_relative_path, expected_matrix)
        actual_matrix = matrix_io.read_matrix(matrix_relative_path)
        self.assertTrue((expected_matrix != actual_matrix).nnz == 0)

    def test_matrix_sbm(self):
        matrix_filename = "test_matrix_sbm.sbm"
        matrix_relative_path = "{}/{}".format(self.TEMP_DIR_NAME, matrix_filename)
        expected_dense_matrix = numpy.random.randint(0, 2, size=(10, 20))
        expected_sparse_matrix = scipy.sparse.coo_matrix(expected_dense_matrix)
        matrix_io.write_matrix(matrix_relative_path, expected_sparse_matrix)
        actual_matrix = matrix_io.read_matrix(matrix_relative_path)
        self.assertTrue((expected_sparse_matrix != actual_matrix).nnz == 0)

    def test_dense_matrix_mtx(self):
        matrix_filename = "test_dense_matrix_mtx.mtx"
        matrix_relative_path = "{}/{}".format(self.TEMP_DIR_NAME, matrix_filename)
        expected_matrix = numpy.random.randn(10, 20)
        matrix_io.write_matrix(matrix_relative_path, expected_matrix)
        actual_matrix = matrix_io.read_matrix(matrix_relative_path)
        self.assertTrue(numpy.allclose(actual_matrix, expected_matrix))

    def test_matrix_sparse_mtx(self):
        matrix_filename = "test_matrix_sparse_mtx.mtx"
        matrix_relative_path = "{}/{}".format(self.TEMP_DIR_NAME, matrix_filename)
        expected_matrix = scipy.sparse.rand(10, 20, 0.5)
        matrix_io.write_matrix(matrix_relative_path, expected_matrix)
        actual_matrix = matrix_io.read_matrix(matrix_relative_path)
        self.assertTrue(numpy.allclose(actual_matrix.todense(), expected_matrix.todense()))

    def test_dense_matrix_csv(self):
        matrix_filename = "test_dense_matrix_csv.csv"
        matrix_relative_path = "{}/{}".format(self.TEMP_DIR_NAME, matrix_filename)
        expected_matrix = numpy.random.randn(10, 20)
        matrix_io.write_matrix(matrix_relative_path, expected_matrix)
        actual_matrix = matrix_io.read_matrix(matrix_relative_path)
        self.assertTrue(numpy.allclose(actual_matrix, expected_matrix))

if __name__ == '__main__':
    unittest.main()
