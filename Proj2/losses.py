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

    def forward(self, target, pred):
 
        self.target = torch.empty((target.shape[0], 2)).scatter_(1, target.view(-1, 1), 1)
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

    def forward(self, target, pred):
 
        self.target = torch.empty((target.shape[0], target.shape[1])).scatter_(1, target.view(-1, 1), 1)
        self.pred = pred.clone()
        return -1 * (self.target * torch.log_softmax(self.pred, dim = 1)).sum() /  target.shape[0]

    def backward(self):
        return torch.softmax(self.pred, dim  = 1) - self.target



