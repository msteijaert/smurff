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
        import StringIO
        return cp.readfp(StringIO.StringIO(str))

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


class TrainStep:
    @classmethod
    def fromStepFile(cls, file_name, iter):
        cp = HeadlessConfigParser(file_name)
        nmodes = int(cp["num_models"])
        step = cls(nmodes, iter)
        step.predictions = pd.read_csv(cp["pred"], sep=";")

        # latent matrices
        for i in range(step.nmodes):
            file_name = cp["model_" + str(i)]
            step.addU(mio.read_matrix(file_name))

        # link matrices (beta)
        for i in range(step.nmodes):
            file_name = cp["prior_" + str(i)]
            try:
                step.add_beta(mio.read_matrix(file_name))
            except FileNotFoundError:
                step.add_beta(np.ndarray((0, 0)))
    
        return step

    def __init__(self, nmodes, iter):
        assert nmodes == 2
        self.nmodes = nmodes
        self.iter = iter
        self.Us = []
        self.betas = []

    def add_beta(self, b):
        self.betas.append(b)

    def addU(self, U):
        self.Us.append(U)

    def num_latent(self):
       return self.Us[0].shape[0]

    def data_shape(self):
       return [ u.shape[1] for u in self.Us ]

    def beta_shape(self):
       return [ b.shape[1] for b in self.betas ]

    def predict_one(self, coords):
        return np.dot(self.Us[0][:,coords[0]], self.Us[1][:,coords[1]])

    def predict_all(self):
        return np.tensordot(self.Us[0],self.Us[1],axes=(0,0))

class PredictSession:
    @classmethod
    def fromRootFile(cls, root_file):
        cp = HeadlessConfigParser(root_file)
        options = OptionsFile(cp["options"])

        session = cls(options.getint("global", "num_priors"))
        for step_name, step_file in cp.items():
            if (step_name.startswith("sample_step")):
                iter = int(step_name[len("sample_step_"):])
                session.addStep(TrainStep.fromStepFile(step_file, iter))

        assert len(session.steps) > 0

        return session

    def __init__(self, nmodes):
        assert nmodes == 2
        self.nmodes = nmodes
        self.steps = []

    def addStep(self, step):
        self.steps.append(step)

    def num_latent(self):
        return self.steps[0].num_latent()

    def data_shape(self):
        return self.steps[0].data_shape()

    def beta_shape(self):
        return self.steps[0].beta_shape()

    def predict_all(self):
        return np.stack([ step.predict_all() for step in self.steps ])
    
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
        return "PredictSession with %d samples\n  Data shape = %s\n  Beta shape = %s\n  Num latent = %d" % (len(self.steps), self.data_shape(), self.beta_shape(), self.num_latent())
