import torch
from torch import optim
from torch import utils
import dlc_practical_prologue as prologue

from Dataset import DataSet

# Only used for performance visualization
import seaborn as sns
import matplotlib.pyplot as plt


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

    # Load into DataSet class
    train_dataset = DataSet(train_input, train_target, train_classes, True)
    test_dataset = DataSet(test_input, test_target, test_classes, False)

    return train_dataset, test_dataset


def compute_nb_errors(model, data_input, data_target, batch_size=100):
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


def train_model(model, train_dataset, val_dataset, criterion, epochs=100, **model_hyperparams):
    '''
    Training model with provided hyperparameters
    '''

    # Use dataloader for shuffling and utilizing data
    train_dataloader = utils.data.DataLoader(
        train_dataset, batch_size=model_hyperparams["batch_size"], shuffle=True)

    device = device_choice()
    model.to(device)
    criterion.to(device)

    # Lists for train loss and accuracy values
    train_losses = torch.zeros(epochs)
    train_acc = torch.zeros(epochs)
    val_acc = torch.zeros(epochs)

    # Scheduler used for learning rate decay: Every 25 epochs lr = lr * gamma
    optimizer = optim.Adam(
        model.parameters(), lr=model_hyperparams["lr"], weight_decay=model_hyperparams["weight_decay"])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
    # * Playable hyperparameters: weight decay, learning rate

    model.train(True)
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_id, train_batch in enumerate(train_dataloader):
            # Auxiliary loss network
            if type(model).__name__ == "AuxiliaryNet":
                optimizer.zero_grad()
                pred, d1, d2 = model(train_batch['input'])

                # Actual objective loss
                loss_main = criterion(pred, train_batch['target'])
                # Auxiliary losses
                loss_aux_1 = criterion(d1, train_batch['class1'].squeeze())
                loss_aux_2 = criterion(d2, train_batch['class2'].squeeze())
                # Final loss is a weighted sum of all losses
                loss = model_hyperparams['aux_param'] * loss_main + (1-model_hyperparams['aux_param'])/2 * \
                    loss_aux_1 + \
                    (1-model_hyperparams['aux_param'])/2 * loss_aux_2

                loss.backward()
                optimizer.step()

            # Convolutional or Dense network
            else:
                optimizer.zero_grad()
                pred = model(train_batch['input'])
                loss = criterion(pred, train_batch['target'])
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()
            # print("Epoch", str(epoch), ", Batch", str(batch_id), " train loss:", str(
            #    loss.item()), "Epoch Loss Summation:", str(epoch_loss))

        # Decay learning rate with scheduler
        scheduler.step()
        print("Epoch", str(epoch), ", Current learning rate:",
              scheduler.get_last_lr(), ", Epoch loss:", epoch_loss)
        train_losses[epoch] = epoch_loss

        train_error = compute_nb_errors(
            model, train_dataset.get_data(), train_dataset.get_target(), batch_size=100)
        train_acc[epoch] = 1-(train_error/train_dataset.get_size())

        val_error = compute_nb_errors(
            model, val_dataset.get_data(), val_dataset.get_target(), batch_size=100)
        val_acc[epoch] = 1-(val_error/val_dataset.get_size())

    model.train(False)

    return train_losses, train_acc, val_acc


def tune_model(model, criterion, epochs=100, rounds=15, **model_hyperparams):
    '''
    This function executes a given model for various hyperparameters and return best set of parameters
    '''

    best_hyperparams = {
        "lr": 0,
        "weight_decay": 0,
        "batch_size": 0,
        "aux_param": 0,
    }
    current_score = -float("inf")

    best_train_loss = torch.zeros(epochs)
    best_train_acc = torch.zeros(epochs)
    best_val_acc = torch.zeros(epochs)

    # Tune learning rate
    for lr_ in model_hyperparams["lr"]:

        # Tune weight decay
        for wd_ in model_hyperparams["weight_decay"]:

            if type(model).__name__ == "AuxiliaryNet":

                # Tune aux params
                for ap_ in model_hyperparams["aux_param"]:

                    avg_train_losses = torch.zeros(epochs)
                    avg_train_acc = torch.zeros(epochs)
                    avg_val_acc = torch.zeros(epochs)

                    # Keep track of the validation accuracy each round
                    rnd_val_acc = torch.zeros(rounds)

                    # Run with newly generated data for 10+ rounds to be sure if training goes well behaved for new data as well
                    for rnd in range(rounds):

                        # Generate dataset
                        (train_input, train_target, train_classes), (val_input, val_target,
                                                                     val_classes) = generate_dataset(model_hyperparams["sample_size"])

                        # Preprocess dataset
                        train_dataset, val_dataset = preprocess_dataset(
                            train_input, train_target, train_classes, val_input, val_target, val_classes)

                        train_losses, train_acc, val_acc = train_model(
                            model, train_dataset, val_dataset, criterion, epochs, lr=lr_, weight_decay=wd_, batch_size=model_hyperparams["batch_size"], aux_param=ap_)

                        avg_train_losses += train_losses
                        avg_train_acc += train_acc
                        avg_val_acc += val_acc

                        rnd_val_acc[rnd] = val_acc[-1]

                    avg_train_losses /= rounds
                    avg_train_acc /= rounds
                    avg_val_acc /= rounds

                    # Get this hyperparam's score
                    score = compute_single_score(rnd_val_acc)

                    # Store the best hyperparams in the hyperparams dict
                    if score > current_score:
                        current_score = score

                        best_train_loss = avg_train_losses.clone()
                        best_train_acc = avg_train_acc.clone()
                        best_val_acc = avg_val_acc.clone()

                        best_hyperparams["lr"] = lr_
                        best_hyperparams["weight_decay"] = wd_
                        best_hyperparams["batch_size"] = model_hyperparams["batch_size"]
                        best_hyperparams["aux_param"] = ap_

            else:

                avg_train_losses = torch.zeros(epochs)
                avg_train_acc = torch.zeros(epochs)
                avg_val_acc = torch.zeros(epochs)

                # Keep track of the validation accuracy each round
                rnd_val_acc = torch.zeros(rounds)

                # Run with newly generated data for 10+ rounds to be sure if training goes well behaved for new data as well
                for rnd in range(rounds):

                    # Generate dataset
                    (train_input, train_target, train_classes), (val_input, val_target,
                                                                 val_classes) = generate_dataset(model_hyperparams["sample_size"])

                    # Preprocess dataset
                    train_dataset, val_dataset = preprocess_dataset(
                        train_input, train_target, train_classes, val_input, val_target, val_classes)

                    train_losses, train_acc, val_acc = train_model(
                        model, train_dataset, val_dataset, criterion, epochs, lr=lr_, weight_decay=wd_, batch_size=model_hyperparams["batch_size"])

                    avg_train_losses += train_losses
                    avg_train_acc += train_acc
                    avg_val_acc += val_acc

                    rnd_val_acc[rnd] = val_acc[-1]

                avg_train_losses /= rounds
                avg_train_acc /= rounds
                avg_val_acc /= rounds

                # Get this hyperparam's score
                score = compute_single_score(rnd_val_acc)

                # Store the best hyperparams in the hyperparams dict
                if score > current_score:
                    current_score = score

                    best_train_loss = avg_train_losses.clone()
                    best_train_acc = avg_train_acc.clone()
                    best_val_acc = avg_val_acc.clone()

                    best_hyperparams["lr"] = lr_
                    best_hyperparams["weight_decay"] = wd_
                    best_hyperparams["batch_size"] = model_hyperparams["batch_size"]

    return best_hyperparams, best_train_loss, best_train_acc, best_val_acc


def compute_single_score(my_tensor):
    '''
    Compute the score from std and mean of a single tensor
    We define the score as mean/std
    '''

    return my_tensor.mean().item()/my_tensor.std().item()


def compute_scores(train_losses, train_acc, val_acc):
    '''
    From the train model, calculate the statistical scores per epoch
    '''

    train_losses_mean = torch.mean(train_losses).item()
    train_losses_std = torch.std(train_losses).item()
    train_losses_min = torch.min(train_losses).item()
    train_losses_max = torch.max(train_losses).item()
    train_losses_median = torch.median(train_losses).item()

    train_acc_mean = torch.mean(train_acc).item()
    train_acc_std = torch.std(train_acc).item()
    train_acc_min = torch.min(train_acc).item()
    train_acc_max = torch.max(train_acc).item()
    train_acc_median = torch.median(train_acc).item()

    val_acc_mean = torch.mean(val_acc).item()
    val_acc_std = torch.std(val_acc).item()
    val_acc_min = torch.min(val_acc).item()
    val_acc_max = torch.max(val_acc).item()
    val_acc_median = torch.median(val_acc).item()

    train_losses_dict = {
        "mean": train_losses_mean,
        "std": train_losses_std,
        "min": train_losses_min,
        "max": train_losses_max,
        "median": train_losses_median
    }

    train_acc_dict = {
        "mean": train_acc_mean,
        "std": train_acc_std,
        "min": train_acc_min,
        "max": train_acc_max,
        "median": train_acc_median,
    }

    val_acc_dict = {
        "mean": val_acc_mean,
        "std": val_acc_std,
        "min": val_acc_min,
        "max": val_acc_max,
        "median": val_acc_median,
    }

    return {
        "train_losses": train_losses_dict,
        "train_acc": train_acc_dict,
        "val_acc": val_acc_dict,
    }


def plot_train_test(train_loss, train_acc, test_acc, model_name):
    '''
    Plot train loss, train accuracy, and test accuracy per epoch on the same graph
    Separate y-axes for loss and accuracy
    '''
    fig, ax1 = plt.subplots(figsize=(8, 8))
    ax1.set_title('Loss and Accuracies: '+model_name, fontsize=16)

    # Plot train loss
    ax1.set_xlabel('Epochs', fontsize=16)
    ax1.set_ylabel('Train loss (log)', fontsize=16)
    ax1.set_yscale("log")
    ax1 = sns.lineplot(x=range(len(train_loss)), y=train_loss,
                       color='tab:red', label='Train loss', legend=False)
    ax1.tick_params(axis='y')
    ax2 = ax1.twinx()

    # Plot train accuracy
    ax2.set_ylabel('Accuracy (%)', fontsize=16)
    ax2 = sns.lineplot(x=range(len(train_acc)), y=train_acc,
                       color='tab:blue', label='Train accuracy', legend=False)
    ax2.tick_params(axis='y')

    # Plot test accuracy
    ax2 = sns.lineplot(x=range(len(test_acc)), y=test_acc,
                       color='tab:green', label='Test accuracy', legend=False)
    ax2.tick_params(axis='y')

    # Set legend
    line1, label1 = ax1.get_legend_handles_labels()
    line2, label2 = ax2.get_legend_handles_labels()
    lines = line1 + line2
    labels = label1 + label2
    ax1.legend(lines, labels, loc='center right')

    plt.savefig('loss_acc_'+model_name.lower() +
                '.png', dpi=800, transparent=True)
    plt.show()


def count_parameters(model):
    '''
    Return the number of trainable parameters in a model
    '''
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
