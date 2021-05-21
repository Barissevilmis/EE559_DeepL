from torch.autograd import grad
from module import Module
from torch import empty
import math


class Linear(Module):
    def __init__(self, *model_params):

        if model_params[0] < 0.0:
            print('Input neuron size must be positive, set to default 20!')
            self.input_neurons = 20
        else:
            self.input_neurons = int(model_params[0])

        if model_params[1] < 0.0:
            print('Output neuron size must be positive, set to default 20!')
            self.output_neurons = 20
        else:
            self.output_neurons = int(model_params[1])

        super().__init__()

        # Always use Xavier initialization
        self.weights = empty(
            (self.input_neurons, self.output_neurons)).uniform_(-1/math.sqrt(self.input_neurons), +1/math.sqrt(self.input_neurons))
        self.bias = empty((1, self.output_neurons)).zero_()

        self.grad_weights = empty((
            self.input_neurons, self.output_neurons)).zero_()
        self.grad_bias = empty((1, self.output_neurons)).zero_()

    def __call__(self, *inpt):
        return self.forward(*inpt)

    def forward(self, *inpt):

        self.data = inpt[0].clone()
        return self.data @ self.weights + self.bias

    def backward(self, *gradwrtoutput):

        self.grad_data = gradwrtoutput[0].clone()
        print(self.grad_weights.shape)
        print(self.data.t().shape)
        print(self.grad_data)
        self.grad_weights += self.data.t() @ self.grad_data
        self.grad_bias += self.grad_data.sum(0)
        return self.grad_data @ self.weights.t()

    def zero_grad(self):
        self.grad_weights = empty(
            self.input_neurons, self.output_neurons).zero_()
        self.grad_bias = empty(1, self.output_neurons).zero_()


class Sequential(Module):
    def __init__(self, *network_structure):
        self.network_structure = network_structure
        super().__init__()

    def __call__(self, *inpt):
        return self.forward(*inpt)

    def forward(self, *inpt):
        data = inpt[0].clone()
        for layer in self.network_structure:
            data = layer.forward(data)

        return data

    def backward(self, *gradwrtoutput):
        data = gradwrtoutput[0].clone()
        for layer in self.network_structure[::-1]:
            data = layer.backward(data)

    def zero_grad(self):
        for layer in self.network_structure[::-1]:
            layer.zero_grad()
