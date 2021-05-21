import torch
import random
from models import Linear, Sequential
from activations import ReLU, Tanh, Sigmoid
from losses import MSE, CrossEntropy
from utils import generate_disc_set, hyperparameter_tuning


train_input, train_target, test_input, test_target = generate_set(1000)


# A distinct model for every different activation function
relu_model = Sequential(Linear(2, 25), ReLU(),
                        Linear(25, 25), ReLU(),
                        Linear(25, 25), ReLU(),
                        Linear(25, 2))

tanh_model = Sequential(Linear(2, 25), Tanh(),
                        Linear(25, 25), Tanh(),
                        Linear(25, 25), Tanh(),
                        Linear(25, 2))

sigmoid_model = Sequential(Linear(2, 25), Sigmoid(),
                            Linear(25, 25), Sigmoid(),
                            Linear(25, 25), Sigmoid(),
                            Linear(25, 2))


