import seaborn as sns
import matplotlib.pyplot as plt

def plot_train_test(loss, accuracy):
    fig, ax1 = plt.subplots(figsize=(8,8))

    # Plot train loss
    ax1.set_title('Training Loss and Test Accuracy', fontsize=16)
    ax1.set_xlabel('Epochs', fontsize=16)
    ax1.set_ylabel('Train loss (log)', fontsize=16)
    ax1.set_yscale("log")
    ax1 = sns.lineplot(x=range(len(loss)), y=loss, color='tab:orange', label='Train loss', legend=False)
    ax1.tick_params(axis='y')
    ax2 = ax1.twinx()

    # Plot test accuracy
    ax2.set_ylabel('Test accuracy (%)', fontsize=16)
    ax2 = sns.lineplot(x=range(len(accuracy)), y=accuracy, color = 'tab:green', label='Test accuracy', legend=False)
    ax2.tick_params(axis='y')

    # Set legend
    line1, label1 = ax1.get_legend_handles_labels()
    line2, label2 = ax2.get_legend_handles_labels()
    lines = line1 + line2
    labels = label1 + label2
    ax1.legend(lines, labels, loc='center right')
    
    plt.savefig('loss_acc.png', dpi=800, transparent=True)
    plt.show()