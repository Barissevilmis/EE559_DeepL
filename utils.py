import torch
import torch.nn as nn
from torch import optim
from torch import utils
import dlc_practical_prologue as prologue

from Dataset import DataSet
from CNN import ConvNet, ConvClassificationNet
from AUXN import AuxiliaryNet
from FC import DenseNet, DenseClassificationNet


def normalize_input(dataset):
    '''
    Normalizes the given dataset in the argument.
    '''
    mean, std = dataset.mean(), dataset.std()
    return (dataset-mean)/std


def device_choice():
    '''
    Choose CPU or GPU as the device
    '''
    if torch.cuda.is_available():
        print("GPU (CUDA) is available!")
        return torch.device("cuda")
    else:
        print("CPU is available!")
        return torch.device("cpu")


def generate_dataset(sample_size=1000):
    '''
    Generate the dataset from prologue file
    '''

    train_input, train_target, train_classes, test_input, test_target, test_classes = prologue.generate_pair_sets(
        sample_size)
    return (train_input, train_target, train_classes), (test_input, test_target, test_classes)


def preprocess_dataset(train_input, train_target, train_classes, test_input, test_target, test_classes):
    '''
    Preprocess train and test datasets
    '''

    # Normalization
    train_input = normalize_input(train_input)
    test_input = normalize_input(test_input)

    train_dataset = DataSet(train_input, train_target, train_classes, True)
    test_dataset = DataSet(test_input, test_target, test_classes, False)


    return train_dataset, test_dataset


def compute_nb_errors(model, data_input, data_target, batch_size=128):
    '''
    Calculate the number of errors given a model, data input, data target and mini batch size.
    Taken from practical exercises
    '''

    nb_data_errors = 0

    for b in range(0, data_input.size(0), batch_size):
        output = model(data_input.narrow(0, b, batch_size))

        pred = d1 = d2 = None

        # If the model is AUXN handle it differently
        if type(model).__name__ == "AuxiliaryNet":
            pred, d1, d2 = output
        else:
            pred = output

        _, predicted_classes = torch.max(pred, 1)
        for k in range(batch_size):
            if data_target[b + k] != predicted_classes[k]:
                nb_data_errors += 1

    return nb_data_errors


def train_model(**model_hyperparams):
    '''
    Training model with provided hyperparameters
    '''

    # Generate dataset
    (train_input, train_target, train_classes), (test_input, test_target, test_classes) = generate_dataset(model_hyperparams["sample_size"])
    # Preprocess dataset
    (train_input, train_target, train_classes), (test_input, test_target, test_classes) = preprocess_dataset(
        (train_input, train_target, train_classes), (test_input, test_target, test_classes))

    #Use dataloader for shuffling and utilizing data
    train_loader = utils.data.DataLoader(train_input, batch_size = model_hyperparams["batch_size"], shuffle = True)
    test_loader = utils.data.DataLoader(test_input, batch_size = model_hyperparams["batch_size"], shuffle = True, drop_last = False)

    #Send model, optimizer and criterion to corresponding device
    model = model_hyperparams['model']
    criterion = model_hyperparams['criterion']

    device = device_choice()
    model.to(device)
    criterion.to(device)

    #Auxiliary loss: Get label digits and use them with auxiliary network
    train_digit_1, train digit_2 = model_hyperparams['train_class'].split(size = 1, dim = 1)
    
    #Lists for train loss and accuracy values
    train_losses = list()
    train_acc = list()

    #Scheduler used for learning rate decay: Every 30 epochs lr = lr * gamma
    optimizer = optim.Adam(model.parameters(), lr = model_hyperparams["lr"], weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)


    for epoch in range(model_hyperparams["epochs"]):
        epoch_loss = 0
        
        for batch_num, batch in enumerate(train_loader):


    

def exec_model(**model_hyperparams):
    '''
    Execute the model by calling train function and compute accuracy values
    '''
    # Train the model and record the train errors at each epoch and accuracy
    train_loss, train_acc = train_model(**model_hyperparams)
    # Calculate errors from training and return pure accuracy values
    train_error = compute_nb_errors(model_hyperparams["model"], model_hyperparams["train_input"],
                                    model_hyperparams["train_target"], model_hyperparams["batch_size"])
    train_acc = 1-(train_error/model_hyperparams["sample_size"])
    test_error = compute_nb_errors(
        model_hyperparams["model"], model_hyperparams["test_input"],
        model_hyperparams["test_target"], model_hyperparams["batch_size"])
    test_acc = 1-(test_error/model_hyperparams["sample_size"])

    return train_acc, test_acc


def tune_model():
    pass


def stats_model():
    pass


if __name__ == "__main__":

    (train_input, train_target, train_classes), (test_input,
                                                 test_target, test_classes) = generate_dataset()
    print(train_input.size())
    print(train_input[0].shape)
    # (train_input, train_target, train_classes), (test_input,
    #                                            test_target, test_classes)=preprocess_dataset(train_input, train_target, train_classes, test_input, test_target, test_classes)
    # print(train_input)