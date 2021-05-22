from losses import MSE
from torch import empty
from utils import compute_nb_errors


class _Optimizer_:
    '''
    Optimizer: - Superclass for Adam and SGD
    Train:     - Training of model by batches for given amount of epochs
               - Zero_grad -> Forward -> Loss -> Backward -> Step
    Choices:   - SGD or Adam
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

    def train(self, train_input, train_target, val_input, val_target):

        train_input_batches = train_input.split(
            split_size=self.batch_size, dim=0)
        train_target_batches = train_target.split(
            split_size=self.batch_size, dim=0)
        val_input_batches = val_input.split(
            split_size=self.batch_size, dim=0)
        val_target_batches = val_target.split(
            split_size=self.batch_size, dim=0)

        epoch_losses = empty((self.epochs)).zero_()
        epoch_losses_val = empty((self.epochs)).zero_()
        train_acc = empty((self.epochs)).zero_()
        val_acc = empty((self.epochs)).zero_()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            epoch_loss_val = 0.0

            for batch_id, (curr_batch, curr_batch_val) in enumerate(zip(train_input_batches, val_input_batches)):
                self.model.zero_grad()

                pred = self.model(curr_batch_val)
                loss = self.criterion(val_target_batches[batch_id], pred)
                epoch_loss_val += loss.item()

                pred = self.model(curr_batch)
                # Calculate loss from criterion with predicted classes and target classes
                loss = self.criterion(train_target_batches[batch_id], pred)
                grad = self.criterion.backward()
                self.model.backward(grad)
                self.step()
                epoch_loss += loss.item()

            val_acc[epoch] = 1 - \
                compute_nb_errors(self.model, val_input, val_target)/1000
            train_acc[epoch] = 1 - \
                compute_nb_errors(self.model, train_input, train_target)/1000
            epoch_losses_val[epoch] = epoch_loss_val
            epoch_losses[epoch] = epoch_loss
            print("Epoch " + str(epoch) + ", Train loss: " + str(epoch_loss))

        return self.model, epoch_losses, epoch_losses_val, train_acc, val_acc

    def step(self):
        pass


class SGDOptimizer(_Optimizer_):
    '''
    Mini batch SGD optimizer: - Only  parameters learning rate
    Learning rate:            - 5e-1 by default
    Optimize by step():       - decrease by learning_rate * gradient
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
        # Inplace modification since mutable objects (i.e. tensors)
        for (w, b, gw, gb) in self.model.param():
            w -= self.lr * gw
            b -= self.lr * gb


class AdamOptimizer(_Optimizer_):
    '''
    Adam optimizer:     - Parameters learning rate, beta1, beta2, epsilon
    Learning rate:      - 1e-3 by default
    Beta1 & Beta2:      - Momentum computation parameters, 0.9 and 0.999 by default 
    Epsilon:            - Final parameter update  -> denominator
    Optimize by step(): - m and v moment updates -> bias corrections -> update weights and bias

    '''

    def __init__(self, model, epochs=100, criterion=MSE(), batch_size=1, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):

        if(lr < 0.0):
            self.lr = 1e-3
            print(
                "Learning rate set to default (1e-3) due to negative learning rate input!")
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
        self.step_size = 1

        super().__init__(model, epochs, criterion, batch_size, lr, 'adam')

    def step(self):

        for (w, b, gw, gb, mw, mb, vw, vb) in self.model.param():

            mw = self.beta1 * mw + (1-self.beta1) * gw.clone()
            mb = self.beta1 * mb + (1-self.beta1) * gb.clone()

            vw = self.beta2 * vw + (1-self.beta2) * (gw.clone() ** 2)
            vb = self.beta2 * vb + (1-self.beta2) * (gb.clone() ** 2)

            bias_corrw1 = mw / (1 - (self.beta1 ** self.step_size))
            bias_corrw2 = vw / (1 - (self.beta2 ** self.step_size))

            bias_corrb1 = mb / (1 - (self.beta1 ** self.step_size))
            bias_corrb2 = vb / (1 - (self.beta2 ** self.step_size))

            w -= self.lr * bias_corrw1 / (bias_corrw2.sqrt() + self.eps)
            b -= self.lr * bias_corrb1 / (bias_corrb2.sqrt() + self.eps)

        self.step_size += 1
