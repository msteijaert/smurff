from .wrapper import PySession

def smurff(train, test = None, **args):
    session = PySession.fromConfig(train, **args)

    session.init()
    while session.step():
        pass

    return session.getResult()

def bpmf(train, test = None, **args):
    args["priors"] = ["normal", "normal"]
    args["side_info_files"] = [ "none", "none" ]
    return smurff(train, **args)

def macau(train, side_info, test = None, **args):
    priors = [ 'normal', 'normal' ]
    for d in range(2):
        if side_info[d] != "none":
            priors[d] = 'macau'

    args["priors"] = priors
    args["side_info_files"] = side_info_files
    args["aux_data"] =  [ [], [] ]

    return smurff(train, **args)

