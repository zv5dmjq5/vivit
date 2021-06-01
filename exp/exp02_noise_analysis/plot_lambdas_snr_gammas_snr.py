"""Generate plots of tuples ``(SNR({λₙₖ}), SNR({γₙₖ}))`` during training."""

import matplotlib.pyplot as plt
import numpy
from shared import CONFIGURATIONS, get_extract_savepath
from shared_plot import get_plot_savepath, get_title, pad_limits, traverse

from exp.utils.path import copy_to_fig, read_from_json
from exp.utils.plot import TikzExport

SUBDIR = "lambdas_snr_gammas_snr"


def get_xlimits(data):
    """Determine shared x limits over all checkpoints."""
    xmin = min(numpy.min(m["lambdas_snr"]) for m in traverse(data))
    xmax = max(numpy.max(m["lambdas_snr"]) for m in traverse(data))

    return xmin, xmax


def get_ylimits(data):
    """Determine shared y limits over all checkpoints."""
    ymin = min(numpy.min(m["gammas_snr"]) for m in traverse(data))
    ymax = max(numpy.max(m["gammas_snr"]) for m in traverse(data))

    return ymin, ymax


def plot(config):
    """Plot SNRs of directional gradients and curvatures."""
    optimizer_cls = config["optimizer_cls"]
    problem_cls = config["problem_cls"]

    loadpath = get_extract_savepath(problem_cls, optimizer_cls)
    data = read_from_json(loadpath)

    xlimits = get_xlimits(data)
    ylimits = get_ylimits(data)

    (xmin, xmax), (ymin, ymax) = pad_limits(xlimits, ylimits)

    for checkpoint, metric in traverse(data, checkpoint=True):
        savepath = get_plot_savepath(checkpoint, problem_cls, optimizer_cls, SUBDIR)

        lambdas_snr = metric["lambdas_snr"]
        gammas_snr = metric["gammas_snr"]

        plt.figure()
        title = (
            get_title(checkpoint, problem_cls, optimizer_cls)
            + f", $K={len(lambdas_snr)}$"
        )
        plt.title(title)

        plt.xlim([xmin, xmax])
        plt.ylim([ymin, ymax])

        plt.xscale("log")
        plt.yscale("log")

        plt.xlabel(r"$\mathrm{SNR}(\lambda_{k})$")
        plt.ylabel(r"$\mathrm{SNR}(\gamma_{k})$")

        plt.plot(lambdas_snr, gammas_snr, marker="o", linestyle="", alpha=0.5)

        TikzExport().save_fig(savepath, tex_preview=False)

        plt.close("all")


def copy(config):
    """Copy files to figure directory."""
    optimizer_cls = config["optimizer_cls"]
    problem_cls = config["problem_cls"]

    loadpath = get_extract_savepath(problem_cls, optimizer_cls)
    data = read_from_json(loadpath)

    src_files = []

    for checkpoint, _ in traverse(data, checkpoint=True):
        src_files.append(
            get_plot_savepath(
                checkpoint, problem_cls, optimizer_cls, SUBDIR, extension=".tex"
            )
        )

    for src in src_files:
        copy_to_fig(src)


if __name__ == "__main__":
    configurations = [config() for config in CONFIGURATIONS]

    for config in configurations:
        plot(config)

    for config in configurations:
        copy(config)
