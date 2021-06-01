"""Experiment utilities for handling paths."""

import os
import shutil

import json_tricks

HERE = os.path.abspath(__file__)
HERE_DIR = os.path.dirname(HERE)

# absolute path to experiment folder
EXP_DIR = os.path.dirname(HERE_DIR)

# absolute path to repository root
REPO_DIR = os.path.dirname(EXP_DIR)

# absolute path to figure folder
_FIG_DIR_NAME = "fig"
FIG_DIR = os.path.join(REPO_DIR, _FIG_DIR_NAME)


def copy_to_fig(src, verbose=True, dry_run=False):
    """Copy a file to the ``fig`` folder. File must be inside ``exp``.

    Preserves the subfolder structure, i.e. the file

    ``exp/exp10_some_name/plots/plot1234.tex``

    will be copied to

    ``fig/exp10_some_name/plots/plot1234.tex``.

    Args:
        src (str, path): Absolute path to the file that will be copied.
        verbose (bool): Print the commands that are being executed.
        dry_run (bool): If ``True``, don't execute the commands.
    """
    src_abs = os.path.abspath(src)

    if os.path.commonpath([src_abs, EXP_DIR]) != EXP_DIR:
        raise ValueError(f"{src_abs} is not in a subdirectory of {EXP_DIR}.")
    if not os.path.exists(src_abs):
        raise ValueError(f"{src_abs} does not exist.")
    if not os.path.isfile(src_abs):
        raise ValueError(f"{src_abs} must be a file.")

    action_str = "would " if dry_run else ""

    dest = os.path.join(FIG_DIR, os.path.relpath(src_abs, start=EXP_DIR))

    # create folders in destination
    dest_dir = os.path.dirname(dest)
    if verbose:
        print(f"[exp] {action_str}create directory (recursively)\n    {dest_dir}")
    if not dry_run:
        os.makedirs(dest_dir, exist_ok=True)

    # copy
    if verbose:
        print(f"[exp] {action_str}copy\n    {src_abs}\n  → {dest}")
    if not dry_run:
        shutil.copy2(src_abs, dest)


def read_from_json(path, verbose=True):
    """Return content of a ``.json`` file.

    Args:
        path (str, Path): Path ending in ``.json`` pointing to the file.
    """
    if verbose:
        print(f"[exp] Reading {path}")

    with open(path) as json_file:
        data = json_tricks.load(json_file)

    return data


def write_to_json(path, data, verbose=True):
    """Write content to a ``.json`` file. Overwrite if file already exists.

    Args:
        path (str, Path): Path ending in ``.json`` pointing to the file.
    """
    if verbose:
        print(f"[exp] Writing to {path}")

    with open(path, "w") as json_file:
        json_tricks.dump(data, json_file)
