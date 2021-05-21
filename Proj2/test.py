from torch import empty
import math
from models import Linear, Sequential
from activations import ReLU, Tanh, Sigmoid
from losses import MSE, CrossEntropy
from utils import generate_set, hyperparameter_tuning


train_input, train_target, test_input, test_target = generate_set(1000)


# A distinct model for every different activation function
relu_model = Sequential(Linear(2, 25), ReLU(),
                        Linear(25, 25), ReLU(),
                        Linear(25, 25), ReLU(),
                        Linear(25, 1), Sigmoid())

tanh_model = Sequential(Linear(2, 25), Tanh(),
                        Linear(25, 25), Tanh(),
                        Linear(25, 25), Tanh(),
                        Linear(25, 1), Sigmoid())

sigmoid_model = Sequential(Linear(2, 25), Sigmoid(),
                           Linear(25, 25), Sigmoid(),
                           Linear(25, 25), Sigmoid(),
                           Linear(25, 1), Sigmoid())
