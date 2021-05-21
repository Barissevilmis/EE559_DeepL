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

model_params_adam = {'lr':[1e-2,1e-3,1e-4], 'beta1': [0.2, 0.6, 0.9], 'beta2': [0.8, 0.9, 0.999], 'weight_decay': [0, 5e-9, 1e-8]}
model_params_sgd = {'lr':[1e-2,1e-3,1e-4,1e-5]}

best_param_adam_relu = hyperparameter_tuning(relu_model, optimizer="adam", criterion=MSE(), epochs=50, batch_size=100, sample_size=1000, rounds=10, **model_params_adam)
best_param_adam_tanh = hyperparameter_tuning(tanh_model, optimizer="adam", criterion=MSE(), epochs=50, batch_size=100, sample_size=1000, rounds=10, **model_params_adam)
best_param_adam_sigmoid = hyperparameter_tuning(sigmoid_model,  optimizer="adam", criterion=MSE(), epochs=50, batch_size=100, sample_size=1000, rounds=10, **model_params_adam)


best_param_sgd_relu = hyperparameter_tuning(relu_model, optimizer="sgd", criterion=MSE(), epochs=50, batch_size=100, sample_size=1000, rounds=10, **model_params_sgd)
best_param_sgd_tanh= hyperparameter_tuning(tanh_model, optimizer="sgd", criterion=MSE(), epochs=50, batch_size=100, sample_size=1000, rounds=10, **model_params_sgd)
best_param_sgd_sigmoid = hyperparameter_tuning(sigmoid_model,  optimizer="sgd", criterion=MSE(), epochs=50, batch_size=100, sample_size=1000, rounds=10, **model_params_sgd)