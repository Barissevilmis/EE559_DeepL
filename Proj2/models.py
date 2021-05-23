from module import Module
from torch import empty
import math


class Linear(Module):
    '''
    This module implements Linear / Dense / Fully Connected layer.
    __init__():  - A default dense layer is set as 20 input and 20 output neurons: User should specify the neuron amount(in + out) as constructer argument
                 - Uniform initialization used as default initializarion
    init_adam(): - SGD or Adam optimizer could be picked as optimizers -> In case of Adam: create variables for m and v moment parameters for both bias and weights
    forward():   - (__call__(): enables model() to be act as model.forward()): f(x_i) = x_i * W + b
    backward():  - Backward propagation: Grad(f(x_i)) = Grad(x_i+1) * W^T
    zero_grad(): - W and b are zeroed out
    param():     - SGD -> Return W, b Grad(W), Grad(b) | Adam -> Return W, b, Grad(W), Grad(b), M_moment(W), M_moment(b), V_moment(W), V_moment(b)
    reset():     - During hyperparameter training, model needs to reset between rounds
    '''
    def __init__(self, *model_params):

        self.model_params = model_params
        self.is_adam = False

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

        # Always use standard uniform initialization
        self.weights = empty(
            (self.input_neurons, self.output_neurons)).uniform_(-1/math.sqrt(self.input_neurons), +1/math.sqrt(self.input_neurons))
        self.bias = empty((1, self.output_neurons)).zero_()

        self.grad_weights = empty((
            self.input_neurons, self.output_neurons)).zero_()
        self.grad_bias = empty((1, self.output_neurons)).zero_()

    def __call__(self, *inpt):
        return self.forward(*inpt)

    def init_adam(self):
        self.is_adam = True
        self.m_moment_weights = empty((
            self.input_neurons, self.output_neurons)).zero_()
        self.v_moment_weights = empty((
            self.input_neurons, self.output_neurons)).zero_()
        self.m_moment_bias = empty((
            1, self.output_neurons)).zero_()
        self.v_moment_bias = empty((
            1, self.output_neurons)).zero_()

    def forward(self, *inpt):

        self.data = inpt[0].clone()
        return self.data @ self.weights + self.bias

    def backward(self, *gradwrtoutput):

        self.grad_data = gradwrtoutput[0].clone()
        self.grad_weights += self.data.t() @ self.grad_data
        self.grad_bias += self.grad_data.sum(0)
        return self.grad_data @ self.weights.t()

    def zero_grad(self):
        self.grad_weights = empty(
            (self.input_neurons, self.output_neurons)).zero_()
        self.grad_bias = empty((1, self.output_neurons)).zero_()

    def param(self):
        res = list()
        if not self.is_adam:
            res.append((self.weights, self.bias,
                       self.grad_weights, self.grad_bias))
        else:
            res.append((self.weights, self.bias, self.grad_weights, self.grad_bias,
                       self.m_moment_weights, self.m_moment_bias, self.v_moment_weights, self.v_moment_bias))
        return res

    def reset(self):
        self.zero_grad()
        self.__init__(*self.model_params)
        if self.is_adam:
            self.init_adam()


class Sequential(Module):
    '''
    This module implements Sequential: Combining and building dense networks .
    __init__():  - Network Structure, layer by layer with activations -> Network merges linear and activations after each other 
    init_adam(): - Activate Adam optimizer over the network
    forward():   - Calls each layers .forward() in incrementing order
    backward():  - Calls each layers .backward() in descending order
    zero_grad(): - Zero_grad called each layer in incrementing order
    param():     - Returns param() for each layer in incrementing order
    reset():     - During hyperparameter training, model needs to reset between rounds: Call each reset within network
    '''
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
        for layer in self.network_structure[::-1]:  # List reversed
            data = layer.backward(data)

    def zero_grad(self):
        for layer in self.network_structure:
            layer.zero_grad()

    def param(self):
        res = list()
        for layer in self.network_structure:
            for param in layer.param():
                if param:
                    res.append(param)
        return res

    def reset(self):
        for layer in self.network_structure:
            layer.reset()
        self.__init__(*self.network_structure)
        return self

    def init_adam(self):
        for layer in self.network_structure:
            layer.init_adam()
