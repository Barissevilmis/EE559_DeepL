import json
import FC
import CNN
import AUXN
import torch
import torch.nn as nn
from utils import count_parameters, tune_model
import hyperparam


criterion = nn.CrossEntropyLoss()
user_input_fc = input("Would you like to tune DenseNet? (Y/N)")
user_input_cnn = input("Would you like to tune ConvNet? (Y/N)")
user_input_aux = input("Would you like to tune AuxiliaryNet? (Y/N)")

if user_input_fc == "Y":
    # Model FC
    model = FC.DenseNet()
    print(f"Number of trainable parameters: {count_parameters(model)}")
    best_hyperparams, best_train_loss, best_train_acc, best_val_acc = tune_model(
        model, criterion, 50, 15, **hyperparam.HYPERPARAMS)
    print(best_hyperparams, best_train_loss, best_train_acc, best_val_acc)
    with open('best_hyperparams_fc.txt', 'w') as file:
        json.dump(best_hyperparams, file)
    torch.save(best_train_loss, 'best_train_loss_fc.pt')
    torch.save(best_train_acc, 'best_train_acc_fc.pt')
    torch.save(best_val_acc, 'best_val_acc_fc.pt')

if user_input_cnn == "Y":
    # Model CNN
    model = CNN.ConvNet()
    print(f"Number of trainable parameters: {count_parameters(model)}")
    best_hyperparams, best_train_loss, best_train_acc, best_val_acc = tune_model(
        model, criterion, 50, 15, **hyperparam.HYPERPARAMS)
    print(best_hyperparams, best_train_loss, best_train_acc, best_val_acc)
    with open('best_hyperparams_cnn.txt', 'w') as file:
        json.dump(best_hyperparams, file)
    torch.save(best_train_loss, 'best_train_loss_cnn.pt')
    torch.save(best_train_acc, 'best_train_acc_cnn.pt')
    torch.save(best_val_acc, 'best_val_acc_cnn.pt')

if user_input_aux == "Y":
    # Model AUXN
    model = AUXN.AuxiliaryNet()
    print(f"Number of trainable parameters: {count_parameters(model)}")
    best_hyperparams, best_train_loss, best_train_acc, best_val_acc = tune_model(
        model, criterion, 50, 15, **hyperparam.HYPERPARAMS)
    print(best_hyperparams, best_train_loss, best_train_acc, best_val_acc)
    with open('best_hyperparams_auxn.txt', 'w') as file:
        json.dump(best_hyperparams, file)
    torch.save(best_train_loss, 'best_train_loss_auxn.pt')
    torch.save(best_train_acc, 'best_train_acc_auxn.pt')
    torch.save(best_val_acc, 'best_val_acc_auxn.pt')
