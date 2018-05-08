#!/usr/bin/env python

from functools import reduce
from math import sqrt
import numpy as np
from scipy import sparse
import scipy.io as sio
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

def read_string(cp,str):
    try:
        return cp.read_string(str)
    except AttributeError:
        from io import StringIO
        return cp.readfp(StringIO(str))

def read_file(cp, file_name):
    with open(file_name) as f:
        try:
            cp.read_file(f, file_name)
        except AttributeError:
            cp.readfp(f, file_name)

class OptionsFile(ConfigParser):
    def __init__(self, file_name):
        ConfigParser.__init__(self) 
        read_file(self, file_name)

class HeadlessConfigParser:
    """A ConfigParser with support for raw items, not in a section"""
    def __init__(self, file_name):
        self.cp = ConfigParser()
        with  open(file_name) as f:
            content = "[top-level]\n" + f.read()
            read_string(self.cp, content)

    def __getitem__(self, key):
        return self.cp.get("top-level", key)

    def items(self):
        return self.cp.items("top-level")

class Sample:
    @classmethod
    def fromStepFile(cls, file_name, iter):
        cp = HeadlessConfigParser(file_name)
        nmodes = int(cp["num_models"])
        sample = cls(nmodes, iter)
        sample.predictions = pd.read_csv(cp["pred"], sep=";")

        # latent matrices
        for i in range(sample.nmodes):
            file_name = cp["model_" + str(i)]
            sample.add_latent(mio.read_matrix(file_name))

        # link matrices (beta)
        for i in range(sample.nmodes):
            file_name = cp["prior_" + str(i)]
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
        self.betas = []

    def check(self):
        for l,b in zip(self.latents, self.betas):
            assert l.shape[0] == self.num_latent()
            assert b.shape[0] == 0 or b.shape[0] == l.shape[1]

    def add_beta(self, b):
        self.betas.append(b)
        self.check()

    def add_latent(self, U):
        self.latents.append(U)
        self.check()

    def num_latent(self):
       return self.latents[0].shape[0]

    def data_shape(self):
       return [ u.shape[1] for u in self.latents ]

    def beta_shape(self):
       return [ b.shape[1] for b in self.betas ]

    def predict_one(self, coords):
        # extract latent vector for each coord
        # we want to call: einsum(U[:,coords[0]], [0], U[:,coords[1]], [0], ...)
        operands = []
        for U,c in zip(self.latents, coords):
            operands += [ U[:,c], [0] ]
        return np.einsum(*operands)

    def predict_all(self):
        # we want to call: einsum(U[0], [0, 0], U[1], [0, 1], U[2], [0, 2], ...)
        operands = []
        for c,l in enumerate(self.latents):
            operands += [l, [0,c+1]]
        return np.einsum(*operands)

    def predict_side(self, mode, side_info):
        ## predict latent vector from side_info 
        uhat = side_info.dot(self.betas[mode].transpose())
        # OTHER modes
        other_modes = list(range(self.nmodes)).remove(mode)
        ## use predicted latent vector to predict activities across columns
        for m in other_modes:
            uhat = uhat.dot(self.latents[m])
        return uhat

class PredictSession:
    @classmethod
    def fromRootFile(cls, root_file):
        cp = HeadlessConfigParser(root_file)
        options = OptionsFile(cp["options"])

        session = cls(options.getint("global", "num_priors"))
        for step_name, step_file in cp.items():
            if (step_name.startswith("sample_step")):
                iter = int(step_name[len("sample_step_"):])
                session.addStep(Sample.fromStepFile(step_file, iter))

        assert len(session.steps) > 0

        return session

    def __init__(self, nmodes):
        assert nmodes == 2
        self.nmodes = nmodes
        self.steps = []

    def addStep(self, sample):
        self.steps.append(sample)

    def num_latent(self):
        return self.steps[0].num_latent()

    def data_shape(self):
        return self.steps[0].data_shape()

    def beta_shape(self):
        return self.steps[0].beta_shape()

    def predict_all(self):
        return np.stack([ sample.predict_all() for sample in self.steps ])
    
    def predict_some(self, test_matrix):
        predictions = Prediction.fromTestMatrix(test_matrix)

        for s in self.steps:
            for p in predictions:
                p.add_sample(s.predict_one(p.coords))

        return predictions

    def predict_one(self, coords, value = float("nan")):
        p = Prediction(coords, value)
        for s in self.steps:
            p.add_sample(s.predict_one(p.coords))

        return p
        
    def __str__(self):
        dat = (len(self.steps), self.data_shape(), self.beta_shape(), self.num_latent())
        return "PredictSession with %d samples\n  Data shape = %s\n  Beta shape = %s\n  Num latent = %d" % dat
