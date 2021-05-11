import torch
import numpy as np
from torch.utils import data
import dlc_practical_prologue as prologue


def normalize_input(dataset):
    """Normalizes the given dataset in the argument.
    """
    mean, std = dataset.mean(), dataset.std()
    return (dataset-mean)/std


def device_choice():
    """Choose CPU or GPU as the device
    """
    if torch.cuda.is_available():
        print("GPU (CUDA) is available!")
        return torch.device("cuda")
    else:
        print("CPU is available!")
        return torch.device("cpu")


def generate_dataset(sample=1000):
    """Generate the dataset from prologue file
    """

    train_input, train_target, train_classes, test_input, test_target, test_classes = prologue.generate_pair_sets(
        sample)
    return (train_input, train_target, train_classes), (test_input, test_target, test_classes)


def preprocess_dataset(train_input, train_target, train_classes, test_input, test_target, test_classes):
    """Preprocess train and test datasets
    """

    # Normalization
    train_input = normalize_input(train_input)
    test_input = normalize_input(test_input)

    # Convert data to float
    train_input = train_input.float()
    train_target = train_target.float()
    train_classes = train_classes.float()
    test_input = test_input.float()
    test_target = test_target.float()
    test_classes = test_classes.float()

    # Set the autograd feature for train and test
    train_input.requires_grad = True
    train_target.requires_grad = True
    train_classes.requires_grad = True
    test_input.requires_grad = False
    test_target.requires_grad = False
    test_classes.requires_grad = False

    # Choose the appropriate device
    device = device_choice()

    # Send the data to the device
    train_input = train_input.to(device)
    train_target = train_target.to(device)
    train_classes = train_classes.to(device)
    test_input = test_input.to(device)
    test_target = test_target.to(device)
    test_classes = test_classes.to(device)

    return (train_input, train_target, train_classes), (test_input, test_target, test_classes)


if __name__ == "__main__":

    (train_input, train_target, train_classes), (test_input,
                                                 test_target, test_classes) = generate_dataset()
    (train_input, train_target, train_classes), (test_input,
                                                 test_target, test_classes) = preprocess_dataset(train_input, train_target, train_classes, test_input, test_target, test_classes)
    print(train_input)
