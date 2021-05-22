import json
from models import Linear, Sequential
from activations import ReLU, Tanh, Sigmoid
from losses import MSE
from utils import hyperparameter_tuning


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

run_sgd = input("Do you want to tune parameters for SGD? (Y/N): ")
run_adam = input("Do you want to tune parameters for Adam? (Y/N): ")

if run_sgd == "Y":

    model_params_sgd = {'lr': [0.9, 0.5, 0.1, 1e-2]}

    best_param_sgd_relu = hyperparameter_tuning(relu_model, optimizer="sgd", criterion=MSE(
    ), epochs=100, batch_size=100, sample_size=1000, rounds=10, **model_params_sgd)

    with open('best_param_sgd_relu.txt', 'w') as file:
        json.dump(best_param_sgd_relu, file)
    best_param_sgd_tanh = hyperparameter_tuning(tanh_model, optimizer="sgd", criterion=MSE(
    ), epochs=100, batch_size=100, sample_size=1000, rounds=10, **model_params_sgd)
    with open('best_param_sgd_tanh.txt', 'w') as file:
        json.dump(best_param_sgd_tanh, file)

    best_param_sgd_sigmoid = hyperparameter_tuning(sigmoid_model,  optimizer="sgd", criterion=MSE(
    ), epochs=100, batch_size=100, sample_size=1000, rounds=10, **model_params_sgd)

    with open('best_param_sgd_sigmoid.txt', 'w') as file:
        json.dump(best_param_sgd_sigmoid, file)

if run_adam == "Y":

    model_params_adam = {
        # 'lr': [0.5, 0.1, 1e-2],
        'lr': [0.9],
        'b1': [0.9, 0.5, 0.2],
        'b2': [0.999, 0.9, 0.5]
    }

    best_param_adam_relu = hyperparameter_tuning(relu_model, optimizer="adam", criterion=MSE(
    ), epochs=100, batch_size=100, sample_size=1000, rounds=10, **model_params_adam)

    with open('best_param_adam_relu.txt', 'w') as file:
        json.dump(best_param_adam_relu, file)
    best_param_adam_tanh = hyperparameter_tuning(tanh_model, optimizer="adam", criterion=MSE(
    ), epochs=100, batch_size=100, sample_size=1000, rounds=10, **model_params_adam)
    with open('best_param_adam_tanh.txt', 'w') as file:
        json.dump(best_param_adam_tanh, file)

    best_param_adam_sigmoid = hyperparameter_tuning(sigmoid_model,  optimizer="adam", criterion=MSE(
    ), epochs=100, batch_size=100, sample_size=1000, rounds=10, **model_params_adam)

    with open('best_param_adam_sigmoid.txt', 'w') as file:
        json.dump(best_param_adam_sigmoid, file)
