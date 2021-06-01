# -*- coding: utf-8 -*-
"""Note: This file exclusively contains functions copied from DeepOBS. The functions 
below are taken from the files ``testproblems_modules.py`` and ``testproblems_utils.py``
"""

from math import ceil

from torch import nn


class net_cifar10_3c3d(nn.Sequential):
    """Basic conv net for cifar10/100.

    The network consists of
    - three conv layers with ReLUs, each followed by max-pooling
    - two fully-connected layers with ``512`` and ``256`` units and ReLU activation
    - output layer with softmax

    The weight matrices are initialized using Xavier initialization and the biases
    are initialized to ``0.0``.
    """

    def __init__(self, num_outputs):
        """Create the net.

        Args:
            num_outputs (int): The numer of outputs (i.e. target classes).
        """
        super(net_cifar10_3c3d, self).__init__()

        self.add_module(
            "conv1", tfconv2d(in_channels=3, out_channels=64, kernel_size=5)
        )
        self.add_module("relu1", nn.ReLU())
        self.add_module(
            "maxpool1",
            tfmaxpool2d(kernel_size=3, stride=2, tf_padding_type="same"),
        )

        self.add_module(
            "conv2", tfconv2d(in_channels=64, out_channels=96, kernel_size=3)
        )
        self.add_module("relu2", nn.ReLU())
        self.add_module(
            "maxpool2",
            tfmaxpool2d(kernel_size=3, stride=2, tf_padding_type="same"),
        )

        self.add_module(
            "conv3",
            tfconv2d(
                in_channels=96,
                out_channels=128,
                kernel_size=3,
                tf_padding_type="same",
            ),
        )
        self.add_module("relu3", nn.ReLU())
        self.add_module(
            "maxpool3",
            tfmaxpool2d(kernel_size=3, stride=2, tf_padding_type="same"),
        )

        self.add_module("flatten", nn.Flatten())

        self.add_module("dense1", nn.Linear(in_features=3 * 3 * 128, out_features=512))
        self.add_module("relu4", nn.ReLU())
        self.add_module("dense2", nn.Linear(in_features=512, out_features=256))
        self.add_module("relu5", nn.ReLU())
        self.add_module("dense3", nn.Linear(in_features=256, out_features=num_outputs))

        # init the layers
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.constant_(module.bias, 0.0)
                nn.init.xavier_normal_(module.weight)

            if isinstance(module, nn.Linear):
                nn.init.constant_(module.bias, 0.0)
                nn.init.xavier_uniform_(module.weight)


def tfmaxpool2d(
    kernel_size,
    stride=None,
    dilation=1,
    return_indices=False,
    ceil_mode=False,
    tf_padding_type=None,
):
    """Implements tf's padding 'same' for maxpooling"""
    modules = []
    if tf_padding_type == "same":
        padding = nn.ZeroPad2d(0)
        hook = hook_factory_tf_padding_same(kernel_size, stride)
        padding.register_forward_pre_hook(hook)
        modules.append(padding)

    modules.append(
        nn.MaxPool2d(
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            return_indices=return_indices,
            ceil_mode=ceil_mode,
        )
    )

    return nn.Sequential(*modules)


def tfconv2d(
    in_channels,
    out_channels,
    kernel_size,
    stride=1,
    dilation=1,
    groups=1,
    bias=True,
    tf_padding_type=None,
):
    modules = []
    if tf_padding_type == "same":
        padding = nn.ZeroPad2d(0)
        hook = hook_factory_tf_padding_same(kernel_size, stride)
        padding.register_forward_pre_hook(hook)
        modules.append(padding)

    modules.append(
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
    )
    return nn.Sequential(*modules)


def hook_factory_tf_padding_same(kernel_size, stride):
    """Generates the torch pre forward hook that needs to be registered on
    the padding layer to mimic tf's padding 'same'"""

    def hook(module, input):
        """The hook overwrites the padding attribute of the padding layer."""
        image_dimensions = input[0].size()[-2:]
        module.padding = _determine_padding_from_tf_same(
            image_dimensions, kernel_size, stride
        )

    return hook


def _determine_padding_from_tf_same(
    input_dimensions, kernel_dimensions, stride_dimensions
):
    """Implements tf's padding 'same' for kernel processes like convolution or pooling.
    Args:
        input_dimensions (int or tuple): dimension of the input image
        kernel_dimensions (int or tuple): dimensions of the convolution kernel
        stride_dimensions (int or tuple): the stride of the convolution

     Returns: A padding 4-tuple for padding layer creation that mimics tf's padding
     'same'.
    """

    # get dimensions
    in_height, in_width = input_dimensions

    if isinstance(kernel_dimensions, int):
        kernel_height = kernel_dimensions
        kernel_width = kernel_dimensions
    else:
        kernel_height, kernel_width = kernel_dimensions

    if isinstance(stride_dimensions, int):
        stride_height = stride_dimensions
        stride_width = stride_dimensions
    else:
        stride_height, stride_width = stride_dimensions

    # determine the output size that is to achive by the padding
    out_height = ceil(in_height / stride_height)
    out_width = ceil(in_width / stride_width)

    # determine the pad size along each dimension
    pad_along_height = max(
        (out_height - 1) * stride_height + kernel_height - in_height, 0
    )
    pad_along_width = max((out_width - 1) * stride_width + kernel_width - in_width, 0)

    # determine padding 4-tuple (can be asymmetric)
    pad_top = pad_along_height // 2
    pad_bottom = pad_along_height - pad_top
    pad_left = pad_along_width // 2
    pad_right = pad_along_width - pad_left

    return pad_left, pad_right, pad_top, pad_bottom
