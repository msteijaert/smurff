import macau
from scipy.sparse import random

Y = random(300,400)

macau.bpmf(Y)
