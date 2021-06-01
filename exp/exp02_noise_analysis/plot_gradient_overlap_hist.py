"""
Generate plots of histogramed (curvature, gradient overlap) tuples
 ``(λₖ, γₖ² / (||g||²)).
"""

import matplotlib.pyplot as plt
import numpy
from shared import CONFIGURATIONS, eigenvalue_cutoff, get_extract_savepath
from shared_plot import get_plot_savepath, get_title, pad_limits, traverse

from exp.utils.path import copy_to_fig, read_from_json
from exp.utils.plot import TikzExport

SUBDIR = "gradient_curvature_overlap_hist"


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

    ymax = 1

    return ymin, ymax


def plot(config, num_bins=50):
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
        trivial_eval = numpy.array([xmin])
        trivial_overlap = numpy.array([max(metric["gradient_trivial_overlap"], ymin)])

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
        plt.ylabel(r"$\mathrm{hist}(\gamma_k^2 / ||g||^2)$")

        # check outliers
        # print(min(lambdas_concat), xmin)
        assert numpy.min(lambdas_concat) >= xmin, "No outliers on the left"
        # print(max(lambdas_concat), xmax)
        assert numpy.max(lambdas_concat) <= xmax, "No outliers on the right"

        bins = numpy.logspace(numpy.log10(xmin), numpy.log10(xmax), num=num_bins)
        plt.hist(
            lambdas_concat,
            log=True,
            bins=bins,
            weights=overlaps_concat,
            color="tab:red",
            bottom=ymin,
        )

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
