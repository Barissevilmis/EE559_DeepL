from module import Module
import torch

class LinearModel(Module):
    def __init__(self, **model_params):
        
        if model_params['input_neurons'] < 0.0:
            print('Input neuron size must be positive, set to default 20!')
            self.input_neurons = 20
        else:
            self.input_neurons = int(model_params['input_neurons'])

        if model_params['output_neurons'] < 0.0:
            print('Output neuron size must be positive, set to default 20!')
            self.output_neurons = 20        
        else:
            self.output_neurons = int(model_params['output_neurons']) 

        super().__init__()

        self.weights = torch.empty((self.input_neurons, self.output_neurons)).uniform_((0,1))
        self.bias = torch.empty((1, self.output_neurons)).normal_((0,1))

        self.grad_








        