#!/usr/bin/env python

from macau import bpmf, macau
import scipy.io

datadir = "/Users/vanderaa/Code/chemogen/data/excape_v3_gfa/compounds_500/"

ic50_train = scipy.io.mmread(datadir + "fold1_train.mtx")
ic50_test = scipy.io.mmread(datadir + "fold1_test.mtx")
ecfp = scipy.io.mmread(datadir + "ecfp.mtx")

result = bpmf(Y          = ic50_train,
              Ytest      = ic50_test,
              num_latent = 20,
              precision   = "adaptive",
              burnin     = 20,
              nsamples   = 100,
              verbose = False)

result = bpmf(Y          = ic50_train,
              Ytest      = ic50_test,
              num_latent = 20,
              verbose    = False,
              precision  = 1.0,
              burnin     = 20,
              nsamples   = 100)

result = macau(Y          = ic50_train,
              Ytest      = ic50_test,
              num_latent = 20,
              side = [ ecfp, None ],
              verbose    = True,
              precision  = 1.0,
              burnin     = 20,
              nsamples   = 100)

