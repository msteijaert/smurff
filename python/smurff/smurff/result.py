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
