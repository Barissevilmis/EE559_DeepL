from torch import empty
import json
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
                        Linear(25, 2), Sigmoid())

tanh_model = Sequential(Linear(2, 25), Tanh(),
                        Linear(25, 25), Tanh(),
                        Linear(25, 25), Tanh(),
                        Linear(25, 2), Sigmoid())

sigmoid_model = Sequential(Linear(2, 25), Sigmoid(),
                           Linear(25, 25), Sigmoid(),
                           Linear(25, 25), Sigmoid(),
                           Linear(25, 2), Sigmoid())


model_params_sgd = {'lr': [1e-0, 5e-1, 1e-1, 5e-2, 1e-2, 5e-3, 1e-3]}


best_param_sgd_relu = hyperparameter_tuning(relu_model, optimizer="sgd", criterion=MSE(
), epochs=50, batch_size=100, sample_size=1000, rounds=10, **model_params_sgd)

with open('best_param_sgd_relu.txt', 'w') as file:
    json.dump(best_param_sgd_relu, file)

best_param_sgd_tanh = hyperparameter_tuning(tanh_model, optimizer="sgd", criterion=MSE(
), epochs=50, batch_size=100, sample_size=1000, rounds=10, **model_params_sgd)

with open('best_param_sgd_tanh.txt', 'w') as file:
    json.dump(best_param_sgd_tanh, file)

best_param_sgd_sigmoid = hyperparameter_tuning(sigmoid_model,  optimizer="sgd", criterion=MSE(
), epochs=50, batch_size=100, sample_size=1000, rounds=10, **model_params_sgd)

with open('best_param_sgd_sigmoid.txt', 'w') as file:
    json.dump(best_param_sgd_sigmoid, file)
