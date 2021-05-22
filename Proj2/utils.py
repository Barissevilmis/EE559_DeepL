from optimizers import SGDOptimizer, AdamOptimizer
from torch import empty
import math
from losses import MSE


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


def hyperparameter_tuning(model, optimizer="sgd", criterion=MSE(), epochs=50, batch_size=100, sample_size=1000, rounds=10, **model_params):

    acc = dict()

    train_acc = empty((rounds)).zero_()
    val_acc = empty((rounds)).zero_()

    if optimizer == "sgd":
        for lr in model_params["lr"]:
            for round in range(rounds):
                print(f"Round {round}:")
                train_input, train_target, test_input, test_target = generate_set(
                    sample_size)
                optim = SGDOptimizer(model.reset(), epochs,
                                     criterion, batch_size, lr=lr)
                trained_model, train_losses = optim.train(
                    train_input, train_target)

                train_acc[round] = 1-compute_nb_errors(
                    trained_model, train_input, train_target) / sample_size
                val_acc[round] = 1-compute_nb_errors(
                    trained_model, test_input, test_target) / sample_size

            acc[lr] = (train_acc.mean(), val_acc.mean(),
                       train_acc.std(), val_acc.std())

        # Pick the best hyperparams
        best_score = -float('inf')
        best_param = None
        for lr in model_params["lr"]:
            (train_acc_mean, val_acc_mean,
             train_acc_std, val_acc_std) = acc[lr]
            if val_acc_mean/val_acc_std > best_score:
                best_score = val_acc_mean/val_acc_std
                best_param = {
                    "lr": lr,
                    "train_acc_mean": train_acc_mean.item(),
                    "val_acc_mean": val_acc_mean.item(),
                    "train_acc_std": train_acc_std.item(),
                    "val_acc_std": val_acc_std.item()
                }

        return best_param

    # Adam
    else:
        for lr in model_params["lr"]:
            for b1 in model_params["b1"]:
                for b2 in model_params["b2"]:
                    for round in range(rounds):
                        print(f"Round {round}:")
                        train_input, train_target, test_input, test_target = generate_set(
                            sample_size)
                        optim = AdamOptimizer(model.reset(), epochs,
                                              criterion, batch_size, lr=lr, beta1=b1, beta2=b2)
                        trained_model, train_losses = optim.train(
                            train_input, train_target)

                        train_acc[round] = 1 - compute_nb_errors(
                            trained_model, train_input, train_target) / sample_size
                        val_acc[round] = 1 - compute_nb_errors(
                            trained_model, test_input, test_target) / sample_size

                    acc[f"{lr},{b1},{b2}"] = (train_acc.mean(), val_acc.mean(),
                                              train_acc.std(), val_acc.std())

        # Pick the best hyperparams
        best_score = -float('inf')
        best_param = None
        for lr in model_params["lr"]:
            for b1 in model_params["b1"]:
                for b2 in model_params["b2"]:
                    (train_acc_mean, val_acc_mean,
                        train_acc_std, val_acc_std) = acc[f"{lr},{b1},{b2}"]
                    if val_acc_mean/val_acc_std > best_score:
                        best_score = val_acc_mean/val_acc_std
                        best_param = {
                            "lr": lr,
                            "b1": b1,
                            "b2": b2,
                            "train_acc_mean": train_acc_mean.item(),
                            "val_acc_mean": val_acc_mean.item(),
                            "train_acc_std": train_acc_std.item(),
                            "val_acc_std": val_acc_std.item()
                        }
        return best_param
