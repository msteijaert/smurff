from .wrapper import PySession

def smurff(Y, **args):
    args["Y"] = Y
    session = PySession.fromConfig(**args)

    session.init()
    while session.step():
        pass

    return session.getResult()

def bpmf(Y, **args):
    args["priors"] = ["normal", "normal"]
    args["side_info_files"] = [ "none", "none" ]
    return smurff(Y, **args)

def macau(train, side_info, **args):
    priors = [ 'normal', 'normal' ]
    for d in range(2):
        if side_info[d] != "none":
            priors[d] = 'macau'

    args["priors"] = priors
    args["side_info_files"] = side_info_files
    args["aux_data"] =  [ [], [] ]
 
    return smurff(**args)

