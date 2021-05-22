import torch
from models import Linear, Sequential
from activations import ReLU, Sigmoid
from losses import MSE
from utils import generate_set
from optimizers import AdamOptimizer

torch.set_grad_enabled(False)


train_input, train_target, test_input, test_target = generate_set(1000)


# A distinct model for every different activation function
relu_model = Sequential(Linear(2, 25), ReLU(),
                        Linear(25, 25), ReLU(),
                        Linear(25, 25), ReLU(),
                        Linear(25, 1), Sigmoid())


optim = AdamOptimizer(relu_model, epochs=100,
                      criterion=MSE(), batch_size=100, lr=0.001, beta1=0.5, beta2=0.9)
_, train_losses, val_losses, train_acc, val_acc = optim.train(
    train_input, train_target, test_input, test_target)

print("\n\n**************FINAL TRAIN AND TEST LOSSES**************\n")

print(f"Final train loss: {train_losses[-1].item()}")
print(f"Final validation loss: {val_losses[-1].item()}")
print(f"Final train accuracy: {train_acc[-1].item()}")
print(f"Final validation accuracy: {val_acc[-1].item()}")

save_params = input("Do you want to save losses and accuracies? (Y/N): ")

if save_params == "Y":
    torch.save(train_losses, 'best_train_losses.pt')
    torch.save(val_losses, 'best_val_losses.pt')
    torch.save(train_acc, 'best_train_acc.pt')
    torch.save(val_acc, 'best_val_acc.pt')
