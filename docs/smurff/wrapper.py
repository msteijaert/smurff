
from .helper import SparseTensor, PyNoiseConfig, StatusItem as PyStatusItem
from .prepare import make_train_test, make_train_test_df
from .result import Prediction
from .predict import PredictSession



class TrainSession:
    """Class for doing a training run in smurff

    A simple use case could be:

    >>> session = smurff.TrainSession(burnin = 5, nsamples = 5)
    >>> session.addTrainAndTest(Ydense)
    >>> session.run()

        
    Attributes
    ----------

    priors: list, where element is one of { "normal", "normalone", "macau", "macauone", "spikeandslab" }
        The type of prior to use for each dimension

    num_latent: int
        Number of latent dimensions in the model

    burnin: int
        Number of burnin samples to discard
    
    nsamples: int
        Number of samples to keep

    num_threads: int
        Number of OpenMP threads to use for model building

    verbose: {0, 1, 2}
        Verbosity level

    seed: float
        Random seed to use for sampling

    save_prefix: path
        Path where to store the samples. The path includes the directory name, as well
        as the initial part of the file names.

    save_freq: int
        - N>0: save every Nth sample
        - N==0: never save a sample
        - N==-1: save only the last sample

    save_extension: { ".csv", ".ddm" }
        - .csv: save in textual csv file format
        - .ddm: save in binary file format

    checkpoint_freq: int
        Save the state of the session every N seconds.

    csv_status: filepath
        Stores limited set of parameters, indicative for training progress in this file. See :class:`StatusItem`

    """
 
    #
    # construction functions
    #
    def __init__(self,
        priors           = [ "normal", "normal" ],
        num_latent       = 16,
        num_threads      = -1,
        burnin           = 400,
        nsamples         = 800,
        seed             = 0,
        verbose          = 1,
        save_prefix      = None,
        save_extension   = None,
        save_freq        = None,
        checkpoint_freq  = None,
        csv_status       = None):

        pass

    def addTrainAndTest(self, Y, Ytest = None, noise = PyNoiseConfig(), is_scarce = True):
        """Adds a train and optionally a test matrix as input data to this TrainSession

        Parameters
        ----------

        Y : :class: `numpy.ndarray`, :mod:`scipy.sparse` matrix or :class: `SparseTensor`
            Train matrix/tensor 
       
        Ytest : :mod:`scipy.sparse` matrix or :class: `SparseTensor`
            Test matrix/tensor. Mainly used for calculating RMSE.

        noise : :class: `PyNoiseConfig`
            Noise model to use for `Y`

        is_scarce : bool
            When `Y` is sparse, and `is_scarce` is *True* the missing values are considered as *unknown*.
            When `Y` is sparse, and `is_scarce` is *False* the missing values are considered as *zero*.
            When `Y` is dense, this parameter is ignored.

        """

        pass

    def addSideInfo(self, mode, Y, noise = PyNoiseConfig(), tol = 1e-6, direct = False):
        """Adds fully known side info, for use in with the macau or macauone prior

        mode : int
            dimension to add side info (rows = 0, cols = 1)

        Y : :class: `numpy.ndarray`, :mod:`scipy.sparse` matrix
            Side info matrix/tensor 
            Y should have as many rows in Y as you have elemnts in the dimension selected using `mode`.
            Columns in Y are features for each element.

        noise : :class: `PyNoiseConfig`
            Noise model to use for `Y`
        
        direct : boolean
            - When True, uses a direct inversion method. 
            - When False, uses a CG solver 

            The direct method is only feasible for a small (< 100K) number of features.

        tol : float
            Tolerance for the CG solver.

        """

        pass

    def addData(self, pos, Y, is_scarce = False, noise = PyNoiseConfig()):
        """Stacks more matrices/tensors next to the main train matrix.

        pos : shape
            Block position of the data with respect to train. The train matrix/tensor
            has implicit block position (0, 0). 

        Y : :class: `numpy.ndarray`, :mod:`scipy.sparse` matrix or :class: `SparseTensor`
            Data matrix/tensor to add

        is_scarce : bool
            When `Y` is sparse, and `is_scarce` is *True* the missing values are considered as *unknown*.
            When `Y` is sparse, and `is_scarce` is *False* the missing values are considered as *zero*.
            When `Y` is dense, this parameter is ignored.

        noise : :class: `PyNoiseConfig`
            Noise model to use for `Y`
        
        """
        pass


    def init(self):
        """Initializes the `TrainSession` after all data has been added.

        You need to call this method befor calling :meth:`step`, unless you call :meth:`run`

        Returns
        -------
        :class:`StatusItem` of the session.

        """

        pass

    def step(self):
        """Does on sampling or burnin iteration.

        Returns
        -------
        - When a step was executed: :class:`StatusItem` of the session.
        - After the last iteration, when no step was executed: `None`.

        """
        pass


    def run(self):
        """Equivalent to:

        .. code-block:: python
        
            self.init()
            while self.step():
                pass
        """

        pass

    def getStatus(self):
        """ Returns :class:`StatusItem` with current state of the session

        """
        
        pass

        
    def getConfig(self):
        """Get this `TrainSession`'s configuration in ini-file format

        """

        pass
  
    def makePredictSession(self):
        """Makes a :class:`PredictSession` based on the model
           that as built in this `TrainSession`.

        """
        pass

    def getTestPredictions(self):
        """Get predictions for test matrix.

        Returns
        -------
        list 
            list of :class:`Prediction`

        """
        pass

    def getRmseAvg(self): 
        """Average RMSE across all samples for the test matrix

        """
        pass
