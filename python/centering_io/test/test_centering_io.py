#!/usr/bin/env python

import unittest
import numpy as np
import centering_io as cio

class TestCenteringIO(unittest.TestCase):
    def test_rows_centering(self):
        origin_matrix = np.array([[1,  2,  3,  4],
                                  [5,  6,  7,  8],
                                  [9, 10, 11, 12]])
        origin_matrix_mean = cio.mean(origin_matrix, 'rows')
        centered_matrix = cio.center(origin_matrix, 'rows', origin_matrix_mean)
        expected_matrix = np.array([[-1.5, -0.5, 0.5, 1.5],
                                    [-1.5, -0.5, 0.5, 1.5],
                                    [-1.5, -0.5, 0.5, 1.5]])
        self.assertTrue(np.allclose(centered_matrix, expected_matrix))

    def test_cols_centering(self):
        origin_matrix = np.array([[1,  2,  3,  4],
                                  [5,  6,  7,  8],
                                  [9, 10, 11, 12]])
        origin_matrix_mean = cio.mean(origin_matrix, 'cols')
        centered_matrix = cio.center(origin_matrix, 'cols', origin_matrix_mean)
        expected_matrix = np.array([[-4,-4, -4, -4],
                                    [ 0, 0,  0,  0],
                                    [ 4, 4,  4,  4]])
        self.assertTrue(np.allclose(centered_matrix, expected_matrix))

    def test_global_centering(self):
        origin_matrix = np.array([[1,  2,  3,  4],
                                  [5,  6,  7,  8],
                                  [9, 10, 11, 12]])
        origin_matrix_mean = cio.mean(origin_matrix, 'global')
        centered_matrix = cio.center(origin_matrix, 'global', origin_matrix_mean)
        expected_matrix = np.array([[-5.5, -4.5, -3.5, -2.5],
                                    [-1.5, -0.5,  0.5,  1.5],
                                    [ 2.5,  3.5,  4.5,  5.5]])
        self.assertTrue(np.allclose(centered_matrix, expected_matrix))

    def test_none_centering(self):
        origin_matrix = np.array([[1,  2,  3,  4],
                                  [5,  6,  7,  8],
                                  [9, 10, 11, 12]])
        origin_matrix_mean = cio.mean(origin_matrix, 'none')
        centered_matrix = cio.center(origin_matrix, 'none', origin_matrix_mean)
        expected_matrix = np.array([[1,  2,  3,  4],
                                    [5,  6,  7,  8],
                                    [9, 10, 11, 12]])
        self.assertTrue(np.allclose(centered_matrix, expected_matrix))

if __name__ == '__main__':
    unittest.main()
