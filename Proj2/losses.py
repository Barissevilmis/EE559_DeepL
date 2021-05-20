from module import Module
import torch

class MSE(Module):
    '''
    MSE: MSE(f(x), y) = sum_i=1^N(f(x_i) - y_i)^2 / N, where N is size
    f(x_i): predictions -> pred
    y_i: true targets -> target
    error_i =  f(x_i) - y_i
    Grad(MSE(f(x), y)) = 2 * sum_i=1^N(error_i)
    '''
    def __init__(self):
        super().__init__()
        self.pred = None
        self.target = None

    def __call__(self, target, pred):
        self.forward(target, pred)
        return self

    def forward(self, target, pred):
 
        self.target = target.clone()
        self.pred = pred.clone()
        error = self.pred - self.target
        return error.pow(2).mean()

    def backward(self):
        error = self.pred - self.target
        return (2 * error) / self.pred.shape[0]


class CrossEntropy(Module):
    '''
    CE: CE(f(x), y) = - sum_i=1^N(I(y_i) * LogSoftmax(f(x_i))) / N, where N is size
    f(x_i): predictions -> pred
    y_i: true targets -> target
    I(y_i) -> Identity one hot vector of y_i target
    error_i = f(x_i) - y_i
    Grad(CE(f(x), y)) = - sum_i=1^N(Softmax(f(x_i)) - I(y_i))
    '''
    def __init__(self):
        super().__init__()
        self.pred = None
        self.target = None

    def __call__(self, target, pred):
        self.forward(target, pred)
        return self

    def forward(self, target, pred):
 
        self.target = target.clone()
        self.pred = pred.clone()
        return -1 * (self.target * self.pred.log_softmax(dim = 1)).sum() /  target.shape[0]

    def backward(self):
        return self.pred.softmax(dim  = 1) - self.target



