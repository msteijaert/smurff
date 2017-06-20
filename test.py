import smurff

print smurff.__version__

from scipy.sparse import random

Y = random(300,400)

print smurff.bpmf(Y, nsamples = 10, burnin = 10, verbose = True)
