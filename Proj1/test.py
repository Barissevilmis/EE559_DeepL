from Dataset import DataSet
import FC
import CNN
import AUXN
import torch.nn as nn
from utils import count_parameters, train_model, generate_dataset

(train_input, train_target, train_classes), (test_input,
                                             test_target, test_classes) = generate_dataset(sample_size=1000)
train_dataset = DataSet(train_input, train_target, train_classes)
test_dataset = DataSet(test_input, test_target, test_classes)
criterion = nn.CrossEntropyLoss()

print("\n\nFully connected dense network:\n\n")
model = FC.DenseNet()
print(f"Number of trainable parameters: {count_parameters(model)}")
model_hyperparams = {"lr": 0.05, "weight_decay": 0.0001,
                     "batch_size": 100, "aux_param": 0}
train_losses, train_acc, val_acc = train_model(model, train_dataset, test_dataset,
                                               criterion, epochs=100, **model_hyperparams)
print(f"Train loss: {train_losses[-1].item()}")
print(f"Train accuracy: {train_acc[-1].item()}")
print(f"Validation accuracy: {val_acc[-1].item()}")

print("\n\nConvolutional neural network:\n\n")
model = CNN.ConvNet()
print(f"Number of trainable parameters: {count_parameters(model)}")
model_hyperparams = {"lr": 0.0005, "weight_decay": 0.001,
                     "batch_size": 100, "aux_param": 0}
train_losses, train_acc, val_acc = train_model(model, train_dataset, test_dataset,
                                               criterion, epochs=100, **model_hyperparams)
print(f"Train loss: {train_losses[-1].item()}")
print(f"Train accuracy: {train_acc[-1].item()}")
print(f"Validation accuracy: {val_acc[-1].item()}")

print("\n\nAuxiliary network:\n\n")
model = AUXN.AuxiliaryNet()
print(f"Number of trainable parameters: {count_parameters(model)}")
model_hyperparams = {"lr": 0.005, "weight_decay": 0.0001,
                     "batch_size": 100, "aux_param": 0.2}
train_losses, train_acc, val_acc = train_model(model, train_dataset, test_dataset,
                                               criterion, epochs=100, **model_hyperparams)
print(f"Train loss: {train_losses[-1].item()}")
print(f"Train accuracy: {train_acc[-1].item()}")
print(f"Validation accuracy: {val_acc[-1].item()}")
