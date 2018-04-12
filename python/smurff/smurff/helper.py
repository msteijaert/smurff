class SparseTensor:
    def __init__(self, shape, data):
        self.shape = shape
        self.ndim = len(shape)
        self.data = data

class PyNoiseConfig:
    def __init__(self, noise_type = "fixed", precision = 5.0, sn_init = 1.0, sn_max = 10.0, threshold = 0.5): 
        self.noise_type = noise_type
        self.precision = precision
        self.sn_init = sn_init
        self.sn_max = sn_max
        self.threshold = threshold

class FixedNoise(PyNoiseConfig):
    def __init__(self, precision = 5.0): 
        PyNoiseConfig.__init__(self, "fixed", precision)

class AdaptiveNoise(PyNoiseConfig):
    def __init__(self, sn_init = 5.0, sn_max = 10.0): 
        PyNoiseConfig.__init__(self, "adaptive", sn_init = sn_init, sn_max = sn_max)

class ProbitNoise(PyNoiseConfig):
    def __init__(self, threshold = 0.): 
        PyNoiseConfig.__init__(self, "probit", threshold = threshold)

