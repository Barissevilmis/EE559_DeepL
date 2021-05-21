from optimizers import AdamOptimizer, SGDOptimizer
from torch import empty
import math
from models import Linear, Sequential
from activations import ReLU, Tanh, Sigmoid
from losses import MSE, CrossEntropy


def generate_set(sample=1000):
    '''
    Generate the train and test dataset. Center of circle=(0.5,0.5), Radius=1/math.sqrt(2*math.pi)
    Return `train_data, train_target, test_data, test_target`
    '''
    # Training set
    train_data = empty(sample, 2).uniform_(0, 1)
    train_target = train_data.sub(0.5).pow(2).sum(1) < 1/(2*math.pi)
    train_target = train_target.int()
    # Testing set
    test_data = empty(sample, 2).uniform_(0, 1)
    test_target = test_data.sub(0.5).pow(2).sum(1) < 1/(2*math.pi)
    test_target = test_target.int()
    return train_data, train_target, test_data, test_target


def compute_nb_errors(model, data_input, data_target, batch_size=100):
    '''
    Calculate the number of errors given a model, data input, data target and mini batch size.
    Taken from practical exercises
    '''

    nb_data_errors = 0

    for b in range(0, data_input.size(0), batch_size):

        pred = model(data_input.narrow(0, b, batch_size))

        for k in range(batch_size):
            if data_target[b + k] != pred[k]:
                nb_data_errors += 1

    return nb_data_errors


def hyperparameter_tuning(model, optimizer="adam", criterion=MSE(), epochs=50, batch_size=100, sample_size=1000, rounds=10, **model_params):

    train_input, train_target, test_input, test_target = generate_set(
        sample_size)
    acc = dict()

    if optimizer == "adam":
        train_acc = empty((rounds)).zero_()
        val_acc = empty((rounds)).zero_()
        for lr in model_params["lr"]:
            for b1 in model_params["beta1"]:
                for b2 in model_params["beta2"]:
                    for wd in model_params["weight_decay"]:
                        for round in range(rounds):
                            optim = AdamOptimizer(
                                model, epochs, criterion, batch_size, lr=lr, beta1=b1, beta2=b2, weight_decay=wd)
                            trained_model, train_losses = optim.train(
                                train_input, train_target)

                            train_acc[round] = compute_nb_errors(
                                trained_model, train_input, train_target) / sample_size
                            val_acc[round] = compute_nb_errors(
                                trained_model, test_input, test_target) / sample_size

                        acc[{"lr": lr, "b1": b1, "b2": b2, "wd": wd}
                            ] = (train_acc.mean(), val_acc.mean(), train_acc.std(), val_acc.std())

        # Pick the best hyperparams
        best_score = -float('inf')
        best_param = None
        for lr in model_params["lr"]:
            for b1 in model_params["beta1"]:
                for b2 in model_params["beta2"]:
                    for wd in model_params["weight_decay"]:
                        (_, val_acc_mean, _, val_acc_std) = acc[{
                            "lr": lr, "b1": b1, "b2": b2, "wd": wd}]
                        if val_acc_mean/val_acc_std > best_score:
                            best_score = val_acc_mean/val_acc_std
                            best_param = {
                                "lr": lr, "b1": b1, "b2": b2, "wd": wd}

        return best_param

    else:
        train_acc = empty((rounds)).zero_()
        val_acc = empty((rounds)).zero_()
        for lr in model_params["lr"]:
            optim = SGDOptimizer(model, epochs, criterion, batch_size, lr=lr)
            trained_model, train_losses = optim.train(train_input, train_target)

            train_acc[round] = compute_nb_errors(
                                trained_model, train_input, train_target) / sample_size
            val_acc[round] = compute_nb_errors(
                                trained_model, test_input, test_target) / sample_size

            acc[{"lr": lr}] = (train_acc.mean(), val_acc.mean(), train_acc.std(), val_acc.std())

         # Pick the best hyperparams
        best_score = -float('inf')
        best_param = None
        for lr in model_params["lr"]:
            (_, val_acc_mean, _, val_acc_std) = acc[{"lr": lr}]
            if val_acc_mean/val_acc_std > best_score:
                best_score = val_acc_mean/val_acc_std
                best_param = {"lr": lr}

        return best_param
