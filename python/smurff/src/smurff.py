import cy_smurff

def smurff(**args):
    session = cy_smurff.createSession(**args)

    while session.step():
        pass

    return session.getResult()

def bpmf(**args):
    args["priors"] = ["normal", "normal"]
    args["side_info_files"] = [ "none", "none" ]
    return smurff(**args)

def macau(side_info, **args):
    priors = [ 'normal', 'normal' ]
    for d in range(2):
        if side_info[d] != "none":
            priors[d] = 'macau'

    args["priors"] = priors
    args["side_info_files"] = side_info_files
    args["aux_data"] =  [ [], [] ]

    return smurff(**args)

