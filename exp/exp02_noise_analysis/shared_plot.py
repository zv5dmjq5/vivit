"""Shared functionality over plotting scripts."""

import os

import numpy
from shared import load_summary

HEREDIR = os.path.dirname(os.path.abspath(__file__))
PLOT_OUTPUT = os.path.join(HEREDIR, "fig")


# IO (plot)
def get_plot_savepath(checkpoint, problem_cls, optimizer_cls, subdir, extension=""):
    """Return save path for metric extraction from noise."""
    savedir = os.path.join(PLOT_OUTPUT, subdir)
    os.makedirs(savedir, exist_ok=True)

    epoch_count, batch_count = [int(count) for count in checkpoint]
    savepath = os.path.join(
        savedir,
        f"{problem_cls.__name__}_{optimizer_cls.__name__}"
        + f"_epoch_{epoch_count:05d}_batch_{batch_count:05d}{extension}",
    )

    return savepath


def get_title(checkpoint, problem_cls, optimizer_cls, detailed=False):
    """Generate title with training statistics."""
    summary = load_summary(problem_cls, optimizer_cls)

    (epoch_count, _) = [int(point) for point in checkpoint]

    train_loss = summary["train_losses"][epoch_count]
    test_loss = summary["test_losses"][epoch_count]

    train_acc_percent = 100 * summary["train_accuracies"][epoch_count]
    test_acc_percent = 100 * summary["test_accuracies"][epoch_count]

    if detailed:
        title = (
            f"Train (test) loss: {train_loss:.3f} ({test_loss:.3f})"
            + f" acc.: {train_acc_percent:.2f}% ({test_acc_percent:.2f}%)"
        )
    else:
        title = f"Loss: {train_loss:.3f}"

    return title


def traverse(metrics, checkpoint=False):
    """Loop through all nestings of saved data, yielding the innermost entries."""
    for epoch_count, epoch_metrics in metrics.items():
        for batch_count, batch_metrics in epoch_metrics.items():

            assert (
                len(batch_metrics.keys()) == 1
            ), f"Only single parameter groups supported: {list(batch_metrics.keys())}"

            for _, group_metrics in batch_metrics.items():
                if checkpoint:
                    yield (epoch_count, batch_count), group_metrics
                else:
                    yield group_metrics


def pad_limits(xlimits, ylimits, padx=0.05, pady=0.05):
    """Add padding around limits in logspace."""

    def pad_log(limits, pad):
        """Add padding in logspace."""
        left, right = limits

        log_left, log_right = numpy.log10(left), numpy.log10(right)
        pad_log_left = log_left - pad * (log_right - log_left)
        pad_log_right = log_right + pad * (log_right - log_left)

        return 10 ** pad_log_left, 10 ** pad_log_right

    xlimits = pad_log(xlimits, padx)
    ylimits = pad_log(ylimits, pady)

    return xlimits, ylimits
