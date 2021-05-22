from torch import empty
import math


def generate_set(sample=1000):
    '''
    Generate the train and test dataset. Center of circle=(0.5,0.5), Radius=1/math.sqrt(2*math.pi)
    Return `train_data, train_target, test_data, test_target`
    '''
    # Training set
    train_data = empty(sample, 2).uniform_(0, 1)
    train_target = train_data.sub(0.5).pow(2).sum(1) < 1/(2*math.pi)
    train_target = train_target.int()
    # Turn the target into shape (1000,1) for compatibility with the final layer
    train_target = train_target[:, None]
    # Testing set
    test_data = empty(sample, 2).uniform_(0, 1)
    test_target = test_data.sub(0.5).pow(2).sum(1) < 1/(2*math.pi)
    test_target = test_target.int()
    # Turn the target into shape (1000,1) for compatibility with the final layer
    test_target = test_target[:, None]
    return train_data, train_target, test_data, test_target


def compute_nb_errors(model, data_input, data_target, batch_size=100):
    '''
    Calculate the number of errors given a model, data input, data target and mini batch size.
    Taken from practical exercises
    '''

    nb_data_errors = 0

    for b in range(0, data_input.size(0), batch_size):

        pred = model(data_input.narrow(0, b, batch_size))

        # Round to nearest integer (0,1)
        predicted_classes = pred.round()
        for k in range(batch_size):
            if data_target[b + k] != predicted_classes[k]:
                nb_data_errors += 1

    return nb_data_errors
