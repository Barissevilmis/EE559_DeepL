class Module(object):

    def __init__(self):
        self.out = None

    def __call__(self):
        raise NotImplementedError

    def forward(self, *inpt):
        raise NotImplementedError

    def backward(self, *gradwrtoutput):
        raise NotImplementedError

    def param(self):
        return []

    def zero_grad(self):
        pass

    def step(self, lr):
        pass
