import FC
import CNN
import AUXN
import torch
import torch.nn as nn
import utils


criterion = nn.CrossEntropyLoss()

# Model FC
model = FC.DenseNet()
print(f"Number of trainable parameters: {utils.count_parameters(model)}")
rnd_val_acc = utils.tune_model_single(
    model, criterion, 50, 15, **{"lr": [0.05], "weight_decay": [0.0001], "batch_size": 100, "aux_param": [0], "sample_size": 1000})
torch.save(rnd_val_acc, 'rnd_val_acc_fc.pt')

# Model CNN
model = CNN.ConvNet()
print(f"Number of trainable parameters: {utils.count_parameters(model)}")
rnd_val_acc = utils.tune_model_single(
    model, criterion, 50, 15, **{"lr": [0.0005], "weight_decay": [0.001], "batch_size": 100, "aux_param": [0], "sample_size": 1000})
torch.save(rnd_val_acc, 'rnd_val_acc_cnn.pt')

# Model AUXN
model = AUXN.AuxiliaryNet()
print(f"Number of trainable parameters: {utils.count_parameters(model)}")
rnd_val_acc = utils.tune_model_single(
    model, criterion, 50, 15, **{"lr": [0.005], "weight_decay": [0.0001], "batch_size": 100, "aux_param": [0.2], "sample_size": 1000})
torch.save(rnd_val_acc, 'rnd_val_acc_auxn.pt')
