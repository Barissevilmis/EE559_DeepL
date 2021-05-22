from module import Module


class ReLU(Module):
    ''' 
    ReLU activation function: f(x_i) = max(0,x_i)
    Forward: f(x_i) = max(0,x_i) return f(x_i)
    Backward: Grad(f(x_i)) = I(x_i > 0.0) * Grad(x_i+1)
    '''

    def __init__(self, ):

        super().__init__()

    def forward(self, *inpt):

        self.out = inpt[0].clone()
        self.out = self.out.relu()
        return self.out

    def backward(self, *gradwrtoutput):

        id = self.out.clone()
        id[id > 0.0] = 1.0
        return id * gradwrtoutput[0].clone()


class Sigmoid(Module):
    ''' 
    Sigmoid activation function: f(x_i) = 1 / (1 + exp(-x_i))
    Forward: f(x_i) = 1 / (1 + exp(-x_i)) return f(x_i)
    Backward: Grad(f(x_i)) = f(x_i) * (1 - f(x_i)) * Grad(x_i+1)
    '''

    def __init__(self):

        super().__init__()

    def forward(self, *inpt):

        self.out = inpt[0].clone()
        self.out = self.out.sigmoid()
        return self.out

    def backward(self, *gradwrtoutput):

        sigmd = self.out.clone()
        grad = sigmd * (1-sigmd)
        return grad * gradwrtoutput[0].clone()


class Tanh(Module):
    ''' 
    Tanh activation function: f(x_i) = (exp(x_i) - exp(-x_i))/ (exp(x_i) + exp(-x_i))
    Forward: f(x_i) = (exp(x_i) - exp(-x_i))/ (exp(x_i) + exp(-x_i)) return f(x_i)
    Backward: Grad(f(x_i)) = (1 - f(x_i)^2) * Grad(x_i+1)
    '''

    def __init__(self):

        super().__init__()

    def forward(self, *inpt):

        self.out = inpt[0].clone()
        self.out = self.out.tanh()
        return self.out

    def backward(self, *gradwrtoutput):

        tanhh = self.out.clone()
        grad = (1 - tanhh.pow(2))
        return grad * gradwrtoutput[0].clone()
