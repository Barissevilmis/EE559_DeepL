import torch
# Only used for performance visualization
import seaborn as sns
import matplotlib.pyplot as plt


def plot_train_test(train_loss, train_acc, test_acc, model_name):
    '''
    Plot train loss, train accuracy, and test accuracy per epoch on the same graph
    Separate y-axes for loss and accuracy
    '''
    fig, ax1 = plt.subplots(figsize=(8, 8))
    ax1.set_title('Loss and Accuracies: '+model_name, fontsize=18)

    # Plot train loss
    ax1.set_xlabel('Epochs', fontsize=18)
    ax1.set_ylabel('Train loss', fontsize=18)
    ax1 = sns.lineplot(x=range(len(train_loss)), y=train_loss,
                       color='tab:red', label='Train loss', legend=False)
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

    # Set accuracy y-axis range
    ax2.set(ylim=(0.7, 1.0))

    fig.tight_layout()
    plt.savefig('loss_acc_'+model_name.lower() +
                '.png', dpi=500, transparent=True)
    plt.show()


def plot_val_scores():
    '''
    Plot validation accuracy scores for each round (15)
    '''
    fc = torch.load('rnd_val_acc_fc.pt')
    cnn = torch.load('rnd_val_acc_cnn.pt')
    auxn = torch.load('rnd_val_acc_auxn.pt')

    print("FC:")
    print(f"std: {fc.std()}, mean: {fc.mean()}")
    print("CNN:")
    print(f"std: {cnn.std()}, mean: {cnn.mean()}")
    print("AUXN:")
    print(f"std: {auxn.std()}, mean: {auxn.mean()}")

    rnds = range(1, 16)

    plt.figure(figsize=(8, 8))
    plt.plot(rnds, fc, label="FC")
    plt.plot(rnds, cnn, label="CNN")
    plt.plot(rnds, auxn, label="AUXN")
    plt.legend(fontsize=16)
    plt.xlabel("Round", fontsize=18)
    plt.ylabel("Final validation accuracy", fontsize=18)
    plt.title("Change of validation accuracy per round", fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    plt.savefig('val_acc_round.png', dpi=500, transparent=True)
    plt.show()


best_train_loss_fc = torch.load("best_train_loss_fc.pt")
best_train_acc_fc = torch.load("best_train_acc_fc.pt")
best_val_acc_fc = torch.load("best_val_acc_fc.pt")

plot_train_test(best_train_loss_fc,
                best_train_acc_fc, best_val_acc_fc, "FC")

best_train_loss_cnn = torch.load("best_train_loss_cnn.pt")
best_train_acc_cnn = torch.load("best_train_acc_cnn.pt")
best_val_acc_cnn = torch.load("best_val_acc_cnn.pt")

plot_train_test(best_train_loss_cnn,
                best_train_acc_cnn, best_val_acc_cnn, "CNN")

best_train_loss_auxn = torch.load("best_train_loss_auxn.pt")
best_train_acc_auxn = torch.load("best_train_acc_auxn.pt")
best_val_acc_auxn = torch.load("best_val_acc_auxn.pt")

plot_train_test(best_train_loss_auxn,
                best_train_acc_auxn, best_val_acc_auxn, "AUXN")

plot_val_scores()
