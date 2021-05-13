import FC
import torch
import torch.nn as nn
import utils
import hyperparam


model = FC.DenseNet()
criterion = nn.CrossEntropyLoss()
best_hyperparams, best_train_loss, best_train_acc, best_val_acc = utils.tune_model(
    model, criterion, 100, 15, **hyperparam.HYPERPARAMS)
print(best_hyperparams, best_train_loss, best_train_acc, best_val_acc)
