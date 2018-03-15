import .cy_smurff 

class ResultItem:
    def __init__(self, coords, val, pred_1sample, pred_avg, var, stds):
        self.coords = coords
        self.val = val
        self.pred_1sample = pred_1sample
        self.pred_avg = pred_avg
        self.var = var
        self.stds = stds

    def __str__(self):
        return "{}: {} | 1sample: {} | avg: {} | var: {} | stds: {}".format(self.coords, self.val, self.pred_1sample, self.pred_avg, self.var, self.stds)

    def __repr__(self):
        return str(self)

class Result:
    def __init__(self, predictions, rmse):
        self.predictions = predictions
        self.rmse = rmse

def smurff(**config_args)
    config = .cy_smurff.createConfig(**config_args)
    session = .cy_smurff.Session.fromConfig(config)

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

