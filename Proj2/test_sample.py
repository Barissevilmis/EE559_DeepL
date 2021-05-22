import torch
from models import Linear, Sequential
from activations import ReLU, Sigmoid
from losses import MSE
from utils import generate_set, compute_nb_errors
from optimizers import SGDOptimizer

torch.set_grad_enabled(False)


train_input, train_target, test_input, test_target = generate_set(1000)


# A distinct model for every different activation function
relu_model = Sequential(Linear(2, 25), ReLU(),
                        Linear(25, 25), ReLU(),
                        Linear(25, 25), ReLU(),
                        Linear(25, 1), Sigmoid())


optim = SGDOptimizer(relu_model, epochs=100,
                     criterion=MSE(), batch_size=100, lr=0.8)
optim.train(train_input, train_target)

err_acc = 1 - compute_nb_errors(
    relu_model, test_input, test_target)/1000
print(f"Test accuracy: {err_acc}")
err_train = 1 - compute_nb_errors(
    relu_model, train_input, train_target)/1000
print(f"Train accuracy: {err_train}")
