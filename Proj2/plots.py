import torch
# Only used for performance visualization
import seaborn as sns
import matplotlib.pyplot as plt


def plot_train_test(train_loss, val_loss, train_acc, test_acc):
    '''
    Plot train loss, train accuracy, and test accuracy per epoch on the same graph
    Separate y-axes for loss and accuracy
    '''
    fig, ax1 = plt.subplots(figsize=(8, 8))
    ax1.set_title('Loss and Accuracies', fontsize=18)

    # Plot train loss
    ax1.set_xlabel('Epochs', fontsize=18)
    ax1.set_ylabel('Loss', fontsize=18)
    ax1 = sns.lineplot(x=range(len(train_loss)), y=train_loss,
                       color='tab:red', label='Train loss', legend=False)
    ax1 = sns.lineplot(x=range(len(val_loss)), y=val_loss,
                       color='tab:orange', label='Test loss', legend=False)
    ax1.tick_params(axis='y')
    ax2 = ax1.twinx()

    # Plot train accuracy
    ax2.set_ylabel('Accuracy', fontsize=18)
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
    ax1.legend(lines, labels, loc='center right', fontsize=16)

    # Set tick font size
    ax1.xaxis.set_tick_params(labelsize=16)
    ax1.yaxis.set_tick_params(labelsize=16)
    ax2.yaxis.set_tick_params(labelsize=16)

    fig.tight_layout()
    plt.savefig('loss_acc.png', dpi=500, transparent=True)
    plt.show()


best_train_loss = torch.load("best_train_losses.pt")
best_val_loss = torch.load("best_val_losses.pt")
best_train_acc = torch.load("best_train_acc.pt")
best_val_acc = torch.load("best_val_acc.pt")

plot_train_test(best_train_loss, best_val_loss,
                best_train_acc, best_val_acc)
