from typing import Optional, Callable
import math
import torch

class _Optimizer_:
    '''
    Optimizer: Superclass for Adam and SGD
    '''
    def __init__(self, model, epochs, criterion, batch_size, lr):

        self.model = model
        self.criterion = criterion
        if epochs <= 0:
            raise ValueError("Epoch must be greater than 0!")
        else: 
            self.epochs = int(epochs)
        if batch_size <= 0:
            raise ValueError("Batch size must be greater than 0!")
        else:
            self.batch_size = int(batch_size)
        if lr <= 0:
            raise ValueError("Learning rate must be greater than 0!")
        else:
            self.lr = float(lr)

    def train(self, train_input, train_target):

        train_input_batches = train_input.split(split_size = self.batch_size,dim = 0)
        train_target_batches = train_target.split(split_size = self.batch_size,dim = 0)

        for epoch in range(self.epochs):
            epoch_loss = 0.0

            for curr_batch in range(0,train_input_batches.shape[0]):
                self.model.zero_grad()

                pred = self.model(train_input_batches[curr_batch])
                loss = self.criterion(train_target_batches[curr_batch],pred)
                grad = self.criterion.backward()
                self.model.backward(grad)
                self.step()

                epoch_loss += loss

            print("Epoch " + str(epoch) + ", Train loss: " + str(epoch_loss))

        return self.model

    def step(self):
        raise NotImplementedError

class AdamOptimizer(_Optimizer_):

    def __init__(self, model, epochs = 50, criterion = MSE(), batch_size = 1, lr = 1e-2, beta1=0.9, beta2=0.999, weight_decay = 0.0, epsilon = 5e-9):

        if(lr < 0.0):
            self.lr = 1e-2
            print("Learning rate set to default (1e-2) due to negative learning rate input!")
        else:
            self.lr = lr
        if(beta1 < 0.0 or beta1 >= 1.0):
            self.beta1 = 0.9
            print("Beta1 set to default value (0.9) due to negative beta1 input!")
        else:
            self.beta1 = beta1
        if(beta2 < 0.0 or beta2 >= 1.0):
            self.beta2 = 0.999
            print("Beta2 set to default value (0.999) due to negative beta2 input!")
        else:
            self.beta2 = beta2
        if(weight_decay < 0.0):
            self.weight_decay = 0.0
            print("Weight decay set to default value (0.0) due to negative weight decay input!")
        else:
            self.weight_decay = weight_decay
        if(epsilon < 0):
            self.epsilon = 5e-9
            print("Epsilon set to default value (5e-9) due to negative input!")
        else:
            self.epsilon = epsilon

        defaults = dict(lr = self.lr, beta1 = self.beta1, beta2=self.beta2, weight_decay=self.weight_decay, epsilon=self.epsilon)

        super().__init__(params, defaults)

    def __initstate__(self):
        """Initialize state variables:
            - there's a 'general' state in which we store the steps
            - there's a state for each model parameter
        """

        for gp in self.param_groups:
            # Initialize parameter-specific state
            for p in gp['params']:
                # But only for parameters having a gradient
                if p.grad is None:
                    continue

                if len(self.state[p]) == 0:
                    self.state[p]['step'] = 0
                    self.state[p]['moment1'] = torch.zeros(p.data.size())
                    self.state[p]['moment2'] = torch.zeros(p.data.size())


    def step(self, closure: Optional[Callable[[], float]] = ...) -> Optional[float]:
        # State stores the variables used by optimization algo
        # need to initialize it if empty
        if len(self.state) == 0:
            self.__initstate__()


        # Iterate over parameter groups
        for pg in self.param_groups:

            #Get all hyper parameters
            lr = pg['lr']
            wd = pg['weight_decay']
            epsilon = pg['epsilon']
            params = pg['params']
            beta1 = pg['beta1']
            beta2 = pg['beta2']

            # Iterate over parameters in the group
            for p in params:

                #Get param within state
                state_p = self.state[p]

                #Increment step by 1
                state_p['step'] += 1
                grad = p.grad.data

                #Only need to continue on parameters with gradient
                if grad is None:
                    continue

                #Adam does not support sparse gradients
                if grad.is_sparse:
                    raise RuntimeError("Adam optimizer does not support sparse gradients!")

                #Bias corrections
                bias1 = 1 - (beta1 ** state_p['step'])
                bias2 = 1 - (beta2 ** state_p['step'])

                #m_t
                state_p['moment1'] = state_p['moment1'] * self.beta1 + (1 - self.beta1) * grad
                #v_t
                state_p['moment2'] = state_p['moment2'] * self.beta2 + grad * grad * (1 - self.beta2)

                #Bias corrections are to be multiplied as they are non-tensor singular values
                bias_corr = math.sqrt(bias2) / bias1
                lr *= bias_corr

                #Update parameters according to bias corrections
                p.data = p.data - lr * state_p['moment1'] / (state_p['moment2'].sqrt() + epsilon)
               
                #If weight decay exists, update parameters accordingly
                if wd != 0.0:
                    p.data = p.data - lr * wd * p.data


class SGDOptimizer(_Optimizer_):
    '''
    Mini batch SGD optimizer: Only  parameters learning rate
    Learning rate: 1e-2 by default
    Optimize by step(): decrease by learning rate * gradient
    '''
    def __init__(self, model, epochs = 100, criterion = MSE(), batch_size = 1, lr = 1e-2):

        if(lr < 0.0):
            self.lr = 1e-2
            print("Learning rate set to default (1e-2) due to negative learning rate input!")
        else:
            self.lr = lr

        if(batch_size < 0):
            self.batch_size = 1
            print("Mini batch size set to default (1) due to negative batch size input!")
        else:
            self.batch_size = batch_size

        defaults = dict(lr = self.lr, batch_size = self.batch_size)

        super(SGDOptimizer,self).__init__(params, defaults)


    def step(self, closure: Optional[Callable[[], float]] = ...) -> Optional[float]:

        # Iterate over parameter groups
        for pg in self.param_groups:

            #Get all hyper parameters
            lr = pg['lr']
            wd = pg['weight_decay']
            params = pg['params']

            # Iterate over parameters in the group
            for p in params:

                #Get param within state
                state_p = self.state[p]

                #Increment step by 1
                state_p['step'] += 1
                grad = p.grad.data

                #Only need to continue on parameters with gradient
                if grad is None:
                    continue

                #Update parameters
                p.data = p.data - lr * grad
               

