import macau

print macau.__version__ 
from scipy.sparse import random

Y = random(300,400)

macau.bpmf(Y, verbose = True)
