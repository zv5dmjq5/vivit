"""Train and save neural networks from the DeepOBS library."""

import os
import warnings

import numpy as np
from deepobs.pytorch.runners import StandardRunner
from shared import CONFIGURATIONS, get_summary_savepath, save_checkpoint_data

from exp.utils.path import write_to_json

HEREDIR = os.path.dirname(os.path.abspath(__file__))
DEEPOBS_OUTPUT = os.path.join(HEREDIR, "results", "deepobs_log")


class CheckpointRunner(StandardRunner):
    """Save model and loss function at checkpoints during training."""

    def training(  # noqa: C901
        self,
        tproblem,
        hyperparams,
        num_epochs,
        print_train_iter,
        train_log_interval,
        tb_log,
        tb_log_dir,
    ):
        """Template training function from ``StandardRunner`` + checkpointing.

        Modified parts are marked by ``CUSTOM``.
        """

        # CUSTOM: Verify all checkpoints will be hit
        self._all_checkpoints_hit(num_epochs)

        opt = self._optimizer_class(tproblem.net.parameters(), **hyperparams)

        # Lists to log train/test loss and accuracy.
        train_losses = []
        valid_losses = []
        test_losses = []
        train_accuracies = []
        valid_accuracies = []
        test_accuracies = []

        minibatch_train_losses = []

        if tb_log:
            try:
                from torch.utils.tensorboard import SummaryWriter

                summary_writer = SummaryWriter(log_dir=tb_log_dir)
            except ImportError as e:
                warnings.warn(
                    "Not possible to use tensorboard for pytorch. Reason: " + e.msg,
                    RuntimeWarning,
                )
                tb_log = False
        global_step = 0

        for epoch_count in range(num_epochs + 1):
            # Evaluate at beginning of epoch.
            self.evaluate_all(
                epoch_count,
                num_epochs,
                tproblem,
                train_losses,
                valid_losses,
                test_losses,
                train_accuracies,
                valid_accuracies,
                test_accuracies,
            )

            # Break from train loop after the last round of evaluation
            if epoch_count == num_epochs:
                break

            ### Training ### # noqa: E266

            # set to training mode
            tproblem.train_init_op()
            batch_count = 0
            while True:
                try:
                    opt.zero_grad()

                    # CUSTOM: Save models at checkpoint
                    if self.is_checkpoint(epoch_count, batch_count):
                        self.checkpoint(epoch_count, batch_count, opt, tproblem)

                    batch_loss, _ = tproblem.get_batch_loss_and_accuracy()
                    batch_loss.backward()
                    opt.step()

                    if batch_count % train_log_interval == 0:
                        minibatch_train_losses.append(batch_loss.item())
                        if print_train_iter:
                            print(
                                "Epoch {0:d}, step {1:d}: loss {2:g}".format(
                                    epoch_count, batch_count, batch_loss
                                )
                            )
                        if tb_log:
                            summary_writer.add_scalar(
                                "loss", batch_loss.item(), global_step
                            )

                    batch_count += 1
                    global_step += 1

                except StopIteration:
                    break

            if not np.isfinite(batch_loss.item()):
                self._abort_routine(
                    epoch_count,
                    num_epochs,
                    train_losses,
                    valid_losses,
                    test_losses,
                    train_accuracies,
                    valid_accuracies,
                    test_accuracies,
                    minibatch_train_losses,
                )
                break
            else:
                continue

        if tb_log:
            summary_writer.close()
        # Put results into output dictionary.
        output = {
            "train_losses": train_losses,
            "valid_losses": valid_losses,
            "test_losses": test_losses,
            "minibatch_train_losses": minibatch_train_losses,
            "train_accuracies": train_accuracies,
            "valid_accuracies": valid_accuracies,
            "test_accuracies": test_accuracies,
        }

        # CUSTOM: Write metrics to a custom file
        self.summary(output, tproblem, opt)

        return output

    # IO helpers

    def set_checkpoints(self, points):
        """Set points where to save the model during training."""
        assert all(len(point) == 2 for point in points), "Checkpoints must be 2-tuples"
        self._checkpoints = [tuple(point) for point in points]

    def get_checkpoints(self):
        """Return checkpoints."""
        try:
            return self._checkpoints
        except AttributeError as e:
            e_msg = getattr(e, "message", repr(e)) + "\nDid you use 'set_checkpoints'?"
            raise Exception(e_msg)

    def is_checkpoint(self, epoch_count, batch_count):
        """Return whether an iteration is a check point."""
        return (epoch_count, batch_count) in self.get_checkpoints()

    def _all_checkpoints_hit(self, num_epochs):
        """Raise exception if checkpoints won't be hit."""
        checkpoint_epochs = [point[0] for point in self._checkpoints]
        assert all(
            epoch < num_epochs for epoch in checkpoint_epochs
        ), "Some checkpoints won't be reached"

    def checkpoint(self, epoch_count, batch_count, opt, tproblem):
        """Save model and loss function at a checkpoint."""
        data = {
            "model": tproblem.net,
            "loss_func": tproblem.loss_function(reduction="mean"),
        }
        save_checkpoint_data(
            data, (epoch_count, batch_count), opt.__class__, tproblem.__class__
        )

    def summary(self, output, tproblem, opt):
        """Save summary."""
        savepath = get_summary_savepath(tproblem.__class__, opt.__class__)
        write_to_json(savepath, output)


def get_runner(config):
    """Set up the DeepOBS runner."""
    optimizer_cls = config["optimizer_cls"]
    hyperparams = config["hyperparams"]

    return CheckpointRunner(optimizer_cls, hyperparams)


def run(config):
    """Train a DeepOBS problem, save model while training."""
    num_epochs = config["num_epochs"]
    batch_size = config["batch_size"]
    checkpoints = config["checkpoints"]
    problem = config["problem_cls"].__name__

    runner = get_runner(config)
    runner.set_checkpoints(checkpoints)

    runner.run(
        testproblem=problem,
        output_dir=DEEPOBS_OUTPUT,
        l2_reg=0.0,
        batch_size=batch_size,
        num_epochs=num_epochs,
        skip_if_exists=True,
    )


if __name__ == "__main__":
    for config in CONFIGURATIONS:
        run(config())
