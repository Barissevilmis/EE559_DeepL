from optimizers import SGDOptimizer
from torch import empty
import torch
import math
from models import Linear, Sequential
from activations import ReLU, Tanh, Sigmoid
from losses import MSE


def generate_set(sample=1000):
    '''
    Generate the train and test dataset. Center of circle=(0.5,0.5), Radius=1/math.sqrt(2*math.pi)
    Return `train_data, train_target, test_data, test_target`
    '''
    # Training set
    train_data = empty(sample, 2).uniform_(0, 1)
    train_target = train_data.sub(0.5).pow(2).sum(1) < 1/(2*math.pi)
    train_target = train_target.int()
    train_target = train_target[:, None]
    # Testing set
    test_data = empty(sample, 2).uniform_(0, 1)
    test_target = test_data.sub(0.5).pow(2).sum(1) < 1/(2*math.pi)
    test_target = test_target.int()
    test_target = test_target[:, None]
    return train_data, train_target, test_data, test_target


def compute_nb_errors(model, data_input, data_target, batch_size=100):
    '''
    Calculate the number of errors given a model, data input, data target and mini batch size.
    Taken from practical exercises
    '''

    nb_data_errors = 0

    for b in range(0, data_input.size(0), batch_size):

        pred = model(data_input.narrow(0, b, batch_size))

        predicted_classes = pred.round()
        for k in range(batch_size):
            if data_target[b + k] != predicted_classes[k]:
                nb_data_errors += 1

    return nb_data_errors


def hyperparameter_tuning(model, optimizer="sgd", criterion=MSE(), epochs=50, batch_size=100, sample_size=1000, rounds=10, **model_params):

    train_input, train_target, test_input, test_target = generate_set(
        sample_size)
    acc = dict()

    train_acc = empty((rounds)).zero_()
    val_acc = empty((rounds)).zero_()
    for lr in model_params["lr"]:
        for round in range(rounds):
            print(f"Round {round}:")
            optim = SGDOptimizer(model.reset(), epochs,
                                 criterion, batch_size, lr=lr)
            trained_model, train_losses = optim.train(
                train_input, train_target)

            train_acc[round] = compute_nb_errors(
                trained_model, train_input, train_target) / sample_size
            val_acc[round] = compute_nb_errors(
                trained_model, test_input, test_target) / sample_size

            acc[lr] = (train_acc.mean(), val_acc.mean(),
                       train_acc.std(), val_acc.std())

        # Pick the best hyperparams
    best_score = -float('inf')
    best_param = None
    for lr in model_params["lr"]:
        (_, val_acc_mean, _, val_acc_std) = acc[lr]
        if val_acc_mean/val_acc_std > best_score:
            best_score = val_acc_mean/val_acc_std
            best_param = {"lr": lr}

    return best_param
