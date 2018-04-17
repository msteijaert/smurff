from scipy import sparse

class Prediction:
    @staticmethod
    def fromTestMatrix(test_matrix):
        return [ Prediction((i, j), v) for i,j,v in zip(*sparse.find(test_matrix)) ]
    
    def __init__(self, coords, val,  pred_1sample = float("nan"), pred_avg = float("nan"), var = float("nan"), iter = -1):
        self.coords = coords
        self.iter = iter
        self.val = val
        self.pred_1sample = pred_1sample
        self.pred_avg = pred_avg
        self.var = var

    def average(self, pred):
        self.iter += 1
        if self.iter == 0:
            self.pred_avg = pred
            self.var = 0
            self.pred_1sample = pred
        else:
            delta = pred - self.pred_avg
            self.pred_avg = (self.pred_avg + delta / (self.iter + 1))
            self.var = self.var + delta * (pred - self.pred_avg)
            self.pred_1sample = pred
            
    def __str__(self):
        return "%s: %.2f | 1sample: %.2f | avg: %.2f | var: %.2f" % (self.coords, self.val, self.pred_1sample, self.pred_avg, self.var)

    def __repr__(self):
        return str(self)

    def __gt__(self, circle2):
        return self.coords > circle2.coords

