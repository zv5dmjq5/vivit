"""Generate plots of (curvature, gradient overlap) tuples ``(λₖ, γₖ² / (||g||²))."""

import matplotlib.pyplot as plt
import numpy
from shared import CONFIGURATIONS, eigenvalue_cutoff, get_extract_savepath
from shared_plot import get_plot_savepath, get_title, pad_limits, traverse

from exp.utils.path import copy_to_fig, read_from_json
from exp.utils.plot import TikzExport

SUBDIR = "gradient_curvature_overlap"


def get_xlimits(data):
    """Determine shared x limits over all checkpoints."""
    xmin = eigenvalue_cutoff
    xmax = max(numpy.max(m["gram_evals"]) for m in traverse(data))

    return xmin, xmax


def get_ylimits(data):
    """Determine shared y limits over all checkpoints."""
    min_overlaps = min(
        numpy.min(m["gradient_curvature_overlap"]) for m in traverse(data)
    )
    min_trivial = min(m["gradient_trivial_overlap"] for m in traverse(data))

    # logscale, no 0
    if min_trivial == 0.0:
        ymin = min_overlaps
    else:
        ymin = min(min_trivial, min_overlaps)

    max_overlaps = max(
        numpy.max(m["gradient_curvature_overlap"]) for m in traverse(data)
    )
    max_trivial = max(m["gradient_trivial_overlap"] for m in traverse(data))

    # logscale, no 0
    if max_trivial == 0.0:
        ymax = max_overlaps
    else:
        ymax = max(max_trivial, max_overlaps)

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

        gram_evals = metric["gram_evals"]
        gradient_overlap = metric["gradient_curvature_overlap"]

        # clip trivial overlap
        trivial_eval = numpy.array([xlimits[0]])
        trivial_overlap = numpy.array(
            [max(metric["gradient_trivial_overlap"], ylimits[0])]
        )

        lambdas_concat = numpy.concatenate((trivial_eval, gram_evals))
        overlaps_concat = numpy.concatenate((trivial_overlap, gradient_overlap))

        plt.figure()
        title = (
            get_title(checkpoint, problem_cls, optimizer_cls)
            + f" (K={len(gram_evals)})$"
        )
        plt.title(title)

        plt.xlim([xmin, xmax])
        plt.ylim([ymin, ymax])

        plt.xscale("log")
        plt.yscale("log")

        plt.xlabel(r"$\lambda_{k}$")
        plt.ylabel(r"$\gamma_k^2 / ||g||^2$")

        plt.plot(lambdas_concat, overlaps_concat, marker="o", linestyle="", alpha=0.5)
        plt.plot(trivial_eval, trivial_overlap, marker="o", linestyle="", alpha=1)

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
