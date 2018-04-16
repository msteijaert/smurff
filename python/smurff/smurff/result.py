from scipy import sparse

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
    def __init__(self, predictions, rmse = float("nan"), iteration = -1):
        if sparse.issparse(predictions):
            test_matrix = predictions
            (I,J,V) = sparse.find(test_matrix)
            self.predictions = []
            for r in range(len(I)):
                r = ResultItem((I[r], J[r]), V[r], .0, .0, .0, .0)
                self.predictions.append(r)

        else:
            self.predictions = predictions

        self.iter = iteration
        self.rmse = rmse
