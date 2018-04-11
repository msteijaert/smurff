from .wrapper import PySession

def smurff(Ytrain, Ytest = None, side_info = None, aux_data = None, **args):
    session = PySession(**args)
    session.addTrainAndTest(Ytrain, Ytest)

    session.init()
    while session.step():
        pass

    return session.getResult()

def bpmf(Y, Ytest = None, **args):
    args["priors"] = ["normal", "normal"]
    return smurff(Y, **args)

def macau(train, side_info, **args):
    priors = [ 'normal', 'normal' ]
    for d in range(2):
        if side_info[d] != "none":
            priors[d] = 'macau'

    args["priors"] = priors
    args["aux_data"] =  [ [], [] ]
 
    return smurff(**args)

