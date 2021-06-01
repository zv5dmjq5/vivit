"""Shared functionality by call scripts among sub-experiments."""

import subprocess

import torch


def bisect(f, left, right):
    """Determine largest value where ``f`` returns ``True``.

    If this values lies to the left of the search interval, return ``left - 1``.
    If this values lies to the right of the search interval, return ``right + 1``.
    """
    if f(left) is False:
        return left - 1
    if f(right) is True:
        return right + 1

    while right - left > 1:
        at = (left + right) // 2

        if f(at):
            left = at
        else:
            right = at

    if f(right):
        return right
    else:
        return left


def run(cmd, show_full_stdout=False, show_full_stderr=False):
    """Execute command as subprocess. Return success."""
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    proc.wait()
    (stdout, stderr) = proc.communicate()

    if proc.returncode == 0:
        print("✔ passing")
        return True
    else:
        last_err = stderr.decode("utf-8").splitlines()
        last_err = last_err[-1] if len(last_err) > 0 else "UNKNOWN"
        print(f"❌ failing with {last_err}")

        if show_full_stdout:
            print("STDOUT:", stdout.decode("utf-8"))
        if show_full_stderr:
            print("STDERR:", stderr.decode("utf-8"))

        return False


def run_batch_size(
    N,
    script,
    device,
    architecture,
    param_groups,
    computations,
    show_full_stdout,
    show_full_stderr,
):
    """Evaluate for batch size, return if successful."""
    print(
        f"\narchitecture = {architecture}\n"
        + f"param_groups = {param_groups}\n"
        + f"computations = {computations}\n"
        + f"device       = {device}\n"
        + f"N = {N:04d}     : ",
        end="",
    )

    cmd = [
        "python",
        script,
        str(N),
        device,
        architecture,
        param_groups,
        computations,
    ]
    success = run(
        cmd, show_full_stdout=show_full_stdout, show_full_stderr=show_full_stderr
    )
    print("\n")

    return success


def get_available_devices(exclude_cpu=False):
    available = []

    if exclude_cpu is False:
        available.append("cpu")

    if torch.cuda.is_available():
        available.append("cuda")

    return available
