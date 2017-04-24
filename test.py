import macau
from scipy.sparse import random

Y = random(300,400)

print macau.bpmf(Y, nsamples = 10, burnin = 10)
