import torch
import torch.nn as nn
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


def generate_dataset(sample_size=1000):
    """Generate the dataset from prologue file
    """

    train_input, train_target, train_classes, test_input, test_target, test_classes = prologue.generate_pair_sets(
        sample_size)
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


def compute_nb_errors(model, data_input, data_target, batch_size=128):
    """Calculate the number of errors given a model, data input, data target and mini batch size.
    Taken from practical exercises
    """

    nb_data_errors = 0

    for b in range(0, data_input.size(0), batch_size):
        output = model(data_input.narrow(0, b, batch_size))

        if type(model).__name__ ==

        _, predicted_classes = torch.max(output, 1)
        for k in range(batch_size):
            if data_target[b + k] != predicted_classes[k]:
                nb_data_errors += 1

    return nb_data_errors


def train_model():
    pass


def exec_model(**model_hyperparams):
    '''Execute the model given model and hyperparameters'''

    # Generate
    (train_input, train_target, train_classes), (test_input, test_target,
                                                 test_classes) = generate_dataset(model_hyperparams["sample_size"])
    # Preprocess
    (train_input, train_target, train_classes), (test_input, test_target, test_classes) = preprocess_dataset(
        (train_input, train_target, train_classes), (test_input, test_target, test_classes))
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
