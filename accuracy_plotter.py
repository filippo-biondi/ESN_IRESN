import matplotlib.pyplot as plt


def accuracy_plotter(result, filename):
    history_dict = result.history
    plot_loss(history_dict, filename)
    plot_accuracy(history_dict, filename)


def plot_loss(history_dict, filename):
    fig, ax = plt.subplots()
    loss_values = history_dict["loss"]
    val_loss_values = history_dict["val_loss"]
    epochs = range(1, len(loss_values) + 1)
    ax.plot(epochs, loss_values, "r", label="Training loss")
    ax.plot(epochs, val_loss_values, "b", label="Validation loss")
    ax.set_title("Training and validation loss")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.legend()
    fig.savefig(filename + "_loss.png")


def plot_accuracy(history_dict, filename):
    fig, ax = plt.subplots()
    acc = history_dict["accuracy"]
    val_acc = history_dict["val_accuracy"]
    epochs = range(1, len(acc) + 1)
    ax.plot(epochs, acc, "g", label="Training acc")
    ax.plot(epochs, val_acc, "y", label="Validation acc")
    ax.set_title("Training and validation accuracy")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Accuracy")
    ax.legend()
    fig.savefig(filename + "_accuracy.png")
