from losses import MSE
from torch import empty


class _Optimizer_:
    '''
    Optimizer: Superclass for Adam and SGD
    '''

    def __init__(self, model, epochs, criterion, batch_size, lr, optim_method):

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

        if optim_method == 'adam':
            self.model.init_adam()

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
                # Calculate loss from criterion with predicted classes and target classes
                loss = self.criterion(train_target_batches[batch_id], pred)
                grad = self.criterion.backward()
                self.model.backward(grad)
                self.step()

                epoch_loss += loss.item()

            epoch_losses[epoch] = epoch_loss
            print("Epoch " + str(epoch) + ", Train loss: " + str(epoch_loss))

        return self.model, epoch_losses

    def step(self):
        pass


class SGDOptimizer(_Optimizer_):
    '''
    Mini batch SGD optimizer: Only  parameters learning rate
    Learning rate: 5e-1 by default
    Optimize by step(): decrease by learning rate * gradient
    '''

    def __init__(self, model, epochs=100, criterion=MSE(), batch_size=1, lr=5e-1):

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

        super().__init__(model, epochs, criterion, batch_size, lr, ['sgd'])

    def step(self):
        for (w, b, gw, gb) in self.model.param():
            w -= self.lr * gw
            b -= self.lr * gb


class AdamOptimizer(_Optimizer_):
    '''
    Adam optimizer: Parameters learning rate, beta1, beta2, weight_decay, epsilon
    Learning rate: 1e-1 by default
    Optimize by step(): decrease by learning rate * gradient
    '''

    def __init__(self, model, epochs=100, criterion=MSE(), batch_size=1, lr=1e-2, beta1=0.9, beta2=0.999, weight_decay=0.0, eps=1e-8):

        if(lr < 0.0):
            self.lr = 1e-2
            print(
                "Learning rate set to default (1e-2) due to negative learning rate input!")
        else:
            self.lr = lr

        if(beta1 < 0.0):
            self.beta1 = 0.9
            print("Beta1 set to default (0.9) due to negative beta1 input!")
        else:
            self.beta1 = beta1

        if(beta2 < 0.0):
            self.beta2 = 0.999
            print("Beta2 set to default (0.999) due to negative beta2 input!")
        else:
            self.beta2 = beta2

        if(weight_decay < 0.0):
            self.weight_decay = 0.0
            print("Weight decay set to default (0.0) due to negative weight decay input!")
        else:
            self.weight_decay = weight_decay

        if(eps < 0.0):
            self.eps = 1e-8
            print("Epsilon set to default (1e-8) due to negative epsilon input!")
        else:
            self.eps = eps

        if(batch_size < 0):
            self.batch_size = 1
            print("Mini batch size set to default (1) due to negative batch size input!")
        else:
            self.batch_size = batch_size
        self.step_size = 0

        super().__init__(model, epochs, criterion, batch_size, lr, 'adam')

    def step(self):

        for (w, b, gw, gb, mw, mb, vw, vb) in self.model.param():

            mw = self.beta1 * mw + (1-self.beta1) * gw.clone()
            mb = self.beta1 * mb + (1-self.beta1) * gb.clone()

            vw = self.beta2 * vw + (1-self.beta2) * (gw.clone() ** 2)
            vb = self.beta2 * vb + (1-self.beta2) * (vb.clone() ** 2)

            mw_corr = mw / (1 - (self.beta1 ** self.step_size))
            mb_corr = mb / (1 - (self.beta1 ** self.step_size))

            vw_corr = vw / (1 - (self.beta2 ** self.step_size))
            vb_corr = vb / (1 - (self.beta2 ** self.step_size))

            w -= ((self.lr * mw_corr) / (vw_corr.sqrt() + self.eps))
            b -= ((self.lr * mb_corr) / (vb_corr.sqrt() + self.eps))

        self.step_size += 1
        self.model.step(self.lr)
