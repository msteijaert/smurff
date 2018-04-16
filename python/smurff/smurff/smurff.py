from .wrapper import TrainSession

def smurff(Ytrain, priors, Ytest = None, side_info = None, **args):
    session = TrainSession(priors = priors, **args)
    session.addTrainAndTest(Ytrain, Ytest)

    if side_info is not None:
        assert len(side_info) == session.nmodes
        for mode in range(session.nmodes):
            si = side_info[mode]
            if si is not None:
                session.addSideInfo(mode, si)

    session.init()
    while session.step():
        pass

    return session.getResult()

def bpmf(Ytrain, Ytest = None, univariate = False ,**args):
    return macau(Ytrain, Ytest, None, univariate, **args)

def macau(Ytrain, Ytest = None, side_info = None, univariate = False, **args):
    nmodes = len(Ytrain.shape)
    priors = ['normal'] * nmodes

    if side_info is not None:
        assert len(side_info) == nmodes
        for d in range(nmodes):
            if side_info[d] is not None:
                priors[d] = 'macau'

    if univariate:
        priors = [ p + "one" for p in priors ]

    return smurff(Ytrain, priors, Ytest, side_info,  **args)

def gfa(Views, Ytest = None, **args):
    Ytrain = Views[0]
    nmodes = len(Ytrain.shape)
    assert nmodes == 2
    priors = ['normal', 'spikeandslab']

    session = TrainSession(priors = priors, **args)
    session.addTrainAndTest(Ytrain, Ytest)

    for p in range(1, len(Views)):
        session.addData([0,p], Views[p])

    session.init()
    while session.step():
        pass

    return session.getResult()

