from activations import ReLU, Tanh, Sigmoid
from models import Linear, Sequential
import json
from optimizers import SGDOptimizer, AdamOptimizer
from torch import empty, set_grad_enabled
from losses import MSE
from utils import generate_set, compute_nb_errors


def hyperparameter_tuning(model, optimizer="sgd", criterion=MSE(), epochs=50, batch_size=100, sample_size=1000, rounds=10, **model_params):

    acc = dict()

    train_acc = empty((rounds)).zero_()
    val_acc = empty((rounds)).zero_()

    if optimizer == "sgd":
        for lr in model_params["lr"]:
            for round in range(rounds):
                print(f"Round {round}:")
                train_input, train_target, test_input, test_target = generate_set(
                    sample_size)
                optim = SGDOptimizer(model.reset(), epochs,
                                     criterion, batch_size, lr=lr)
                trained_model, _, _, _, _ = optim.train(
                    train_input, train_target, test_input, test_target)

                train_acc[round] = 1-compute_nb_errors(
                    trained_model, train_input, train_target) / sample_size
                val_acc[round] = 1-compute_nb_errors(
                    trained_model, test_input, test_target) / sample_size

            acc[lr] = (train_acc.mean(), val_acc.mean(),
                       train_acc.std(), val_acc.std())

        # Pick the best hyperparams
        best_score = -float('inf')
        best_param = None
        for lr in model_params["lr"]:
            (train_acc_mean, val_acc_mean,
             train_acc_std, val_acc_std) = acc[lr]
            if val_acc_mean/val_acc_std > best_score:
                best_score = val_acc_mean/val_acc_std
                best_param = {
                    "lr": lr,
                    "train_acc_mean": train_acc_mean.item(),
                    "val_acc_mean": val_acc_mean.item(),
                    "train_acc_std": train_acc_std.item(),
                    "val_acc_std": val_acc_std.item()
                }

        return best_param

    # Adam
    else:
        for lr in model_params["lr"]:
            for b1 in model_params["b1"]:
                for b2 in model_params["b2"]:
                    for round in range(rounds):
                        print(f"Round {round}:")
                        train_input, train_target, test_input, test_target = generate_set(
                            sample_size)
                        optim = AdamOptimizer(model.reset(), epochs,
                                              criterion, batch_size, lr=lr, beta1=b1, beta2=b2)
                        trained_model, _, _, _, _ = optim.train(
                            train_input, train_target, test_input, test_target)

                        train_acc[round] = 1 - compute_nb_errors(
                            trained_model, train_input, train_target) / sample_size
                        val_acc[round] = 1 - compute_nb_errors(
                            trained_model, test_input, test_target) / sample_size

                    acc[f"{lr},{b1},{b2}"] = (train_acc.mean(), val_acc.mean(),
                                              train_acc.std(), val_acc.std())

        # Pick the best hyperparams
        best_score = -float('inf')
        best_param = None
        for lr in model_params["lr"]:
            for b1 in model_params["b1"]:
                for b2 in model_params["b2"]:
                    (train_acc_mean, val_acc_mean,
                        train_acc_std, val_acc_std) = acc[f"{lr},{b1},{b2}"]
                    if val_acc_mean/val_acc_std > best_score:
                        best_score = val_acc_mean/val_acc_std
                        best_param = {
                            "lr": lr,
                            "b1": b1,
                            "b2": b2,
                            "train_acc_mean": train_acc_mean.item(),
                            "val_acc_mean": val_acc_mean.item(),
                            "train_acc_std": train_acc_std.item(),
                            "val_acc_std": val_acc_std.item()
                        }
        return best_param


set_grad_enabled(False)


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
        'lr': [1e-1, 1e-2, 1e-3],
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


if __name__ == "__main__":

    set_grad_enabled(False)

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
            'lr': [1e-1, 1e-2, 1e-3],
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
