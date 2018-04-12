from .wrapper import PySession

def smurff(Ytrain, Ytest = None, side_info = None, **args):
    session = PySession(**args)
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

def bpmf(Y, Ytest = None, **args):
    return macau(Y, Ytest, **args)

def macau(train, Ytest = None, side_info = None, **args):
    nmodes = len(train.shape)

    priors = ['normal'] * nmodes
    if side_info is not None:
        assert len(side_info) == session.nmodes
        for d in range(nmodes):
            if side_info[d] is not None:
                priors[d] = 'macau'


    return smurff(Ytrain, Ytest, side_info, priors, **args)

