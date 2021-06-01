"""Generate train/test accuracy plots."""

import matplotlib.pyplot as plt
from shared import CONFIGURATIONS, load_summary
from shared_plot import get_plot_savepath

from exp.utils.path import copy_to_fig
from exp.utils.plot import TikzExport

SUBDIR = "accuracy"


def plot(config):
    """Plot train and test accuracy."""
    optimizer_cls = config["optimizer_cls"]
    problem_cls = config["problem_cls"]
    num_epochs = config["num_epochs"]

    summary = load_summary(problem_cls, optimizer_cls)

    epochs = list(range(num_epochs + 1))

    train_acc_percent = [100 * val for val in summary["train_accuracies"]]
    test_acc_percent = [100 * val for val in summary["test_accuracies"]]

    checkpoint = (num_epochs - 1, 0)
    savepath = get_plot_savepath(checkpoint, problem_cls, optimizer_cls, SUBDIR)

    plt.figure()

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.plot(epochs, train_acc_percent, linestyle="-", label="Train")
    plt.plot(epochs, test_acc_percent, linestyle="--", label="Test")
    plt.legend()

    TikzExport().save_fig(savepath, tex_preview=False)

    plt.close("all")


def copy(config):
    """Copy files to figure directory."""
    optimizer_cls = config["optimizer_cls"]
    problem_cls = config["problem_cls"]
    num_epochs = config["num_epochs"]

    checkpoint = (num_epochs - 1, 0)
    savepath = get_plot_savepath(
        checkpoint, problem_cls, optimizer_cls, SUBDIR, extension=".tex"
    )

    copy_to_fig(savepath)


if __name__ == "__main__":
    configurations = [config() for config in CONFIGURATIONS]

    for config in configurations:
        plot(config)

    for config in configurations:
        copy(config)
