from typing import Optional, Callable
from losses import MSE, CrossEntropy
import math
from torch import empty


class _Optimizer_:
    '''
    Optimizer: Superclass for Adam and SGD
    '''

    def __init__(self, model, epochs, criterion, batch_size, lr):

        self.model = model
        self.criterion = criterion
        self.epochs = None
        self.batch_size = None
        self.lr = None
        if epochs <= 0:
            print("Epoch must be greater than 0, set to default of 50!")
            self.epochs = 50
        else:
            self.epochs = int(epochs)
        if batch_size <= 0:
            print("Batch size must be greater than 0, set to default 32!")
            self.batch_size = 32
        else:
            self.batch_size = int(batch_size)
        if lr <= 0:
            print("Learning rate must be greater than 0, set to default 1e-2!")
            self.lr = 1e-2
        else:
            self.lr = float(lr)

    def train(self, train_input, train_target):

        train_input_batches = train_input.split(
            split_size=self.batch_size, dim=0)
        train_target_batches = train_target.split(
            split_size=self.batch_size, dim=0)

        epoch_losses = empty((self.epochs)).zero_()
        for epoch in range(self.epochs):
            epoch_loss = 0.0

            for batch_id, curr_batch in enumerate(train_input_batches):
                self.model.zero_grad()

                pred = self.model(curr_batch)
                loss = self.criterion(curr_batch, pred)
                grad = self.criterion.backward()
                self.model.backward(grad)
                self.step()

                epoch_loss += loss.item()

            epoch_losses[epoch] = epoch_loss
            print("Epoch " + str(epoch) + ", Train loss: " + str(epoch_loss))

        return self.model, epoch_losses

    def step(self):
        raise NotImplementedError


class SGDOptimizer(_Optimizer_):
    '''
    Mini batch SGD optimizer: Only  parameters learning rate
    Learning rate: 1e-2 by default
    Optimize by step(): decrease by learning rate * gradient
    '''

    def __init__(self, model, epochs=100, criterion=MSE(), batch_size=1, lr=1e-2):

        if(lr < 0.0):
            self.lr = 1e-2
            print(
                "Learning rate set to default (1e-2) due to negative learning rate input!")
        else:
            self.lr = lr

        if(batch_size < 0):
            self.batch_size = 1
            print("Mini batch size set to default (1) due to negative batch size input!")
        else:
            self.batch_size = batch_size

        super().__init__(model, epochs, criterion, batch_size, lr)

    def step(self, closure: Optional[Callable[[], float]] = ...) -> Optional[float]:

        # Iterate over parameter groups
        self.model.step(self.lr)
