#!/usr/bin/env python

import numpy as np
from scipy import sparse
import pandas as pd
import os
import os.path
import matrix_io as mio
from glob import glob
import re
import csv

try:
    from configparser import ConfigParser
except ImportError:
    from ConfigParser import SafeConfigParser as ConfigParser

from .result import Prediction

def read_config_file(file_name):
    cp = ConfigParser()

    with open(file_name) as f:
        try:
            cp.read_file(f, file_name)
        except AttributeError:
            cp.readfp(f, file_name)

    return cp

class Sample:
    @classmethod
    def fromStepFile(cls, file_name, iter):
        cp = read_config_file(file_name)
        nmodes = int(cp["models"]["num_models"])
        sample = cls(nmodes, iter)

        # latent matrices
        for i in range(sample.nmodes):
            file_name = cp["models"]["model_" + str(i)]
            sample.add_latent(mio.read_matrix(file_name))

        # link matrices (beta)
        for i in range(sample.nmodes):
            file_name = cp["priors"]["prior_" + str(i)]
            try:
                sample.add_beta(mio.read_matrix(file_name))
            except FileNotFoundError:
                sample.add_beta(np.ndarray((0, 0)))

        return sample

    def __init__(self, nmodes, iter):
        assert nmodes == 2
        self.nmodes = nmodes
        self.iter = iter
        self.latents = []
        self.latent_means = []
        self.betas = []

    def check(self):
        for l, b in zip(self.latents, self.betas):
            assert l.shape[0] == self.num_latent()
            assert b.shape[0] == 0 or b.shape[0] == self.num_latent()

    def add_beta(self, b):
        self.betas.append(b)
        self.check()

    def add_latent(self, U):
        self.latents.append(U)
        self.latent_means.append(np.mean(U, axis=1))
        self.check()

    def num_latent(self):
       return self.latents[0].shape[0]

    def data_shape(self):
       return [u.shape[1] for u in self.latents]

    def beta_shape(self):
       return [b.shape[1] for b in self.betas]

    def predict(self, coords_or_sideinfo=None):
        # for one prediction: einsum(U[:,coords[0]], [0], U[:,coords[1]], [0], ...)
        # for all predictions: einsum(U[0], [0, 0], U[1], [0, 1], U[2], [0, 2], ...)

        cs = coords_or_sideinfo if coords_or_sideinfo is not None else [
            None] * self.nmodes

        operands = []
        for U, Umean, c, m in zip(self.latents, self.latent_means, cs, range(self.nmodes)):
            # predict all in this dimension
            if c is None:
                operands += [U, [0, m+1]]
            else:
                # if side_info was specified for this dimension, we predict for this side_info
                try:  # try to compute sideinfo * beta using dot
                    # compute latent vector from side_info
                    uhat = c.dot(self.betas[m].transpose()) + Umean
                    uhat = np.squeeze(uhat)
                    operands += [uhat, [0]]
                except AttributeError:  # assume it is a coord
                    # if coords was specified for this dimension, we predict for this coord
                    operands += [U[:, c], [0]]

        return np.einsum(*operands)


class PredictSession:
    """Session for making predictions using a model generated using a :class:`TrainSession`.

    A :class:`PredictSession` can be made directly from a :class:`TrainSession`

    >>> predict_session  = train_session.makePredictSession()

    or from a root file

    >>> predict_session = PredictSession.fromRootFile("save-root.ini")

    """
    @classmethod
    def fromRootFile(cls, root_file):
        """Creates a :class:`PredictSession` from a give root file
 
        Parameters
        ----------
        root_file : string
           Name of the root file.
 
        """
        cp = read_config_file(root_file)
        options = read_config_file(cp["options"]["options"])

        session = cls(options.getint("global", "num_priors"))
        for step_name, step_file in cp["steps"].items():
            if (step_name.startswith("sample_step")):
                iter = int(step_name[len("sample_step_"):])
                session.add_sample(Sample.fromStepFile(step_file, iter))

        assert len(session.samples) > 0

        return session

    def __init__(self, nmodes):
        assert nmodes == 2
        self.nmodes = nmodes
        self.samples = []

    def add_sample(self, sample):
        self.samples.append(sample)

    def num_latent(self):
        return self.samples[0].num_latent()

    def data_shape(self):
        return self.samples[0].data_shape()

    def beta_shape(self):
        return self.samples[0].beta_shape()

    def predict(self, coords_or_sideinfo=None):
        return np.stack([sample.predict(coords_or_sideinfo) for sample in self.samples])

    def predict_all(self):
        """Computes the full prediction matrix/tensor.

        Returns
        -------
        A :class:`numpy.ndarray` of shape `[ N x T1 x T2 x ... ]` where
        N is the number of samples in this `PredictSession` and `T1 x T2 x ...` 
        is the shape of the train data.

        """        
        return self.predict()

    def predict_some(self, test_matrix):
        """Computes prediction for all elements in a sparse test matrix

        Parameters
        ----------
        test_matrix : scipy sparse matrix
            Coordinates and true values to make predictions for

        Returns
        -------
        A list of :class:`Prediction` objects.

        """        
        predictions = Prediction.fromTestMatrix(test_matrix)

        for s in self.samples:
            for p in predictions:
                p.add_sample(s.predict(p.coords))

        return predictions

    def predict_one(self, coords, value=float("nan")):
        """Computes prediction for one point in the matrix/tensor

        Parameters
        ----------
        coords : shape
            The position of the desired point
        value : float, optional
            The *true* value for this point

        Returns
        -------
        A :class:`Prediction` object

        """
        p = Prediction(coords, value)
        for s in self.samples:
            p.add_sample(s.predict(p.coords))

        return p

    def __str__(self):
        dat = (len(self.samples), self.data_shape(),
               self.beta_shape(), self.num_latent())
        return "PredictSession with %d samples\n  Data shape = %s\n  Beta shape = %s\n  Num latent = %d" % dat
