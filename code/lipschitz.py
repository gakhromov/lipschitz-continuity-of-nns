"""This module contains the essential Lipschitz estimation functions."""
import warnings
from collections.abc import Callable

import numpy as np
import scipy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet
from utils.conv_to_matrix import get_conv2d_matrix
from utils.functions import (check_basic_block_structure,
                             check_bottleneck_structure)
from utils.math import (compute_matrix_1norm, compute_matrix_1norm_batched,
                        compute_matrix_2norm_power_method,
                        compute_matrix_2norm_power_method_batched)


def compute_lower_bound(
    model: torch.nn.Module,
    dataset_dataloader: DataLoader,
    device: torch.device = torch.device("cpu"),
    norm_ord: int | float = 2,
) -> torch.Tensor:
    """Computes the lower bound of the whole model, given the dataset.

    Parameters
    ----------
    model
        The model object.
    dataset_dataloader
        Dataset dataloader, that will be used for computing the lower bound.
    device, optional
        Torch device, by default torch.device("cpu").
    norm_ord, optional
        Matrix or vector norm to use for Jacobian computation.
        Be careful to user the appropriate norm to obtain proper Lipschitz 
        bounds. By default 2.

    Returns
    -------
        Two tensors with one number, representing the lower Lipschitz and the 
        average Lipschitz bounds.
    """
    n_samples = len(dataset_dataloader.dataset)

    lower_bound = 0
    avg_bound = 0

    for i, (x_batch, y_batch) in enumerate(dataset_dataloader):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        norms = compute_jac_norm(model, x_batch, ord=norm_ord)

        lower = torch.max(norms)
        if lower > lower_bound:
            lower_bound = lower

        avg_bound += torch.sum(norms)

    avg_bound /= n_samples
    return lower_bound, avg_bound


def compute_upper_bound(
    lipschitz_per_layer: float | list | dict | torch.Tensor,
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Computes the upper bound of the whole model, given all Lipschitz 
    constants for each layer.

    Parameters
    ----------
    lipschitz_per_layer
        Output of the `compute_lipschitz_upper_bound_per_layer` function, 
        can be a Tensor, float, list or a dict.
    dtype, optional
        dtype to use for Lipschitz computation, by default torch.float32.

    Returns
    -------
        A tensor with one number, representing the answer.

    Raises
    ------
    TypeError
        This exception is raised when `lipschitz_per_layer` type does not match 
        torch.Tensor, dict or list.

    """
    if isinstance(lipschitz_per_layer, torch.Tensor):
        return lipschitz_per_layer.type(dtype)

    if isinstance(lipschitz_per_layer, float):
        return torch.tensor(lipschitz_per_layer, dtype=dtype)

    if isinstance(lipschitz_per_layer, dict):
        # handle residual layers
        residual = compute_upper_bound(lipschitz_per_layer["residual"], dtype)
        sequential = compute_upper_bound(
            lipschitz_per_layer["sequential"], dtype)
        return residual + sequential

    if isinstance(lipschitz_per_layer, list):
        # compute the product of sequential layers
        return torch.stack([
            compute_upper_bound(i, dtype) for i in lipschitz_per_layer]).prod()

    raise TypeError(f"Lipschitz per layer type {type(lipschitz_per_layer)} is "
                    "unknown for this function.")


def compute_lipschitz_upper_bound_per_layer(
    layer: torch.nn.Module,
    layer_input_shape: list[int],
    dtype: torch.dtype = torch.float32,
    ord: int | float = 2,
) -> torch.Tensor | list | dict:
    """Returns the Lipschitz constant for each particular layer of the model.
    If a Sequence is given, outputs a list of Lipschitz constants. If a 
    Bottleneck or BasicBlock are given, output a dictionary with a sequence of 
    Lipschitz constants for the sequential and residual parts.

    Parameters
    ----------
    layer
        Layer object.
    layer_input_shape
        Shape(s) of the input for this layer. Must be present to compute Conv 
        layers' Lip. constant.
    dtype
        dtype to use for Lipschitz computation, by default torch.float32.
    ord, optional
        The order of the norm, by default 2. Possible options: 1, 2, torch.inf.

    Returns
    -------
        The Lipschitz constant for this particular layer (not considering 
        previous layers).
    """
    # supress warnings from scipy
    warnings.filterwarnings(action="ignore", module="scipy")

    if isinstance(layer, nn.Sequential):
        # recursive call for nested sequential layers
        return [
            compute_lipschitz_upper_bound_per_layer(
                layer[i], layer_input_shape[i], dtype)
            for i in range(len(layer))
        ]

    if isinstance(layer, nn.Linear):
        # for linear layers, jacobian is the weight matrix
        w = layer.state_dict()["weight"]

        if ord == 2:
            return compute_matrix_2norm_power_method(w).type(dtype)
        if ord == 1:
            return compute_matrix_1norm(w).type(dtype)
        if ord == torch.inf:
            return torch.linalg.matrix_norm(w, ord=torch.inf).type(dtype)

    if isinstance(layer, nn.Conv2d):
        # Here, we compute the lipschitz constant of the layer as a
        # largest singular value of the identical linear matrix multiplication.

        conv_kernel = layer.weight.cpu().detach().numpy()
        img_size = layer_input_shape[-1]
        K = get_conv2d_matrix(
            conv_kernel, layer.padding[0], layer.stride[0], img_size)

        if ord == 2:
            return torch.tensor(scipy.sparse.linalg.norm(K, ord=2), dtype=dtype)
        if ord == 1:
            return np.abs(K).sum(axis=0).max(axis=1)
        if ord == torch.inf:
            return np.abs(K).sum(axis=1).max(axis=0)

    if isinstance(layer, nn.BatchNorm2d):
        state = layer.state_dict()
        var = state["running_var"]
        gamma = state["weight"]
        eps = layer.eps

        return torch.max(torch.abs(gamma / torch.sqrt(var + eps))).type(dtype)

    if isinstance(layer, resnet.BasicBlock) or isinstance(layer, resnet.Bottleneck):
        # this is a ResNet residual block
        if isinstance(layer, resnet.BasicBlock):
            assert check_basic_block_structure(layer)
        else:
            assert check_bottleneck_structure(layer)

        # in comments starting with # //, we present a forward pass of the
        # BasicBlock acc. to
        # https://pytorch.org/vision/stable/_modules/torchvision/models/resnet.html
        # for clarity

        # // identity = x
        lip_residual = torch.tensor(1.0, dtype=dtype)

        # // out = self.conv1(x)
        lip = [compute_lipschitz_upper_bound_per_layer(
            layer.conv1, layer_input_shape[0], dtype)]
        # // out = self.bn1(out)
        lip += [compute_lipschitz_upper_bound_per_layer(layer.bn1, [], dtype)]
        # // out = self.relu(out)

        # // out = self.conv2(out)
        lip += [compute_lipschitz_upper_bound_per_layer(
            layer.conv2, layer_input_shape[1], dtype)]
        # // out = self.bn2(out)
        lip += [compute_lipschitz_upper_bound_per_layer(layer.bn2, [], dtype)]

        # in case of "Bottleneck" module
        if isinstance(layer, resnet.Bottleneck):
            # // out = self.relu(out)

            # // out = self.conv3(out)
            lip += [
                compute_lipschitz_upper_bound_per_layer(
                    layer.conv3, layer_input_shape[2], dtype)
            ]
            # // out = self.bn3(out)
            lip += [compute_lipschitz_upper_bound_per_layer(
                layer.bn3, [], dtype)]

        # // if self.downsample is not None:
        # //     identity = self.downsample(x)
        if layer.downsample is not None:
            # downsample is a sequential layer with a convolution and batchnorm
            lip_residual = compute_lipschitz_upper_bound_per_layer(
                layer.downsample, [layer_input_shape[0], []], dtype
            )

        # // out += identity
        # // out = self.relu(out)
        return {"residual": lip_residual, "sequential": lip}

    # by default, for other types of layers, output 1
    # (be careful to include all non 1-Lipschitz layers in the check before)
    return torch.tensor(1.0, dtype=dtype)


def compute_jac_norm(model: Callable,
                     x_batch: torch.Tensor,
                     ord: int | float = 2) -> torch.Tensor:
    """Computes the norm of the model jacobian for each input in the batch.
    This method works for all models with any inputs of any dimension. The 
    jacobian matrix is represented as a (num_outputs x flat_num_inputs) matrix.

    Parameters
    ----------
    model
        Model function.
    x_batch
        Batched inputs tensor of shape (batch_size x input_shape).
        Input_shape can be a number or 3 numbers (channels x height x width), 
        making x_batch of shape (batch_size x channels x height x width).
    ord, optional
        The order of the norm, by default 2. Possible options: 1, 2, torch.inf.

    Returns
    -------
        A vector of norms of the jacobian for each input in the batch.
    """
    x_batch.requires_grad_(True)
    out = model(x_batch)

    # jacobian of dimensions batch_size x output_dim x input_dim
    # input_dim can be a list
    jac = compute_jacobian(out, x_batch)

    # to compute the 2-norm, we flatten the jacobian in the dim. of the input
    jac = jac.flatten(start_dim=2, end_dim=-1)

    if ord == 1:
        jac_norm = compute_matrix_1norm_batched(jac)
    if ord == 2:
        jac_norm = compute_matrix_2norm_power_method_batched(jac)
    if ord == torch.inf:
        jac_norm = torch.linalg.matrix_norm(jac, ord=torch.inf)

    x_batch.requires_grad_(False)

    return jac_norm


@torch.jit.script
def compute_jacobian(
    y_batch: torch.Tensor, x_batch: torch.Tensor, create_graph: bool = False
) -> torch.Tensor:
    """Computes the Jacobian of y wrt to x.
    Thanks to https://github.com/magamba/overparameterization/blob/530948c72662b062fcb0c5c084b857a3951efb63/core/metric_helpers.py#L235
    for providing the code to this function.

    Parameters
    ----------
    y_batch
        Output tensor of the model, batched. Dimensions: batch_size x output_dim
    x_batch
        Input tensor of the model, batched. Dimensions: batch_size x input_dim
    create_graph, optional
        Flag that tells torch to create the graph. Useful to compute hessians. 
        By default False.

    Returns
    -------
        Jacobian of y_batch wrt to x_batch. 
        Dimensions: batch_size x output_dim x input_dim.

    Note
    ----
        x_batch has to track gradients before the function is called. If you use
        this function for second order derivatives, RESET THE GRADIENT OF 
        x_batch BY x_batch.grad = None. Otherwise, the second derivative will be
        added to the first derivative. If there are other nodes in the graph 
        that depend on y_batch, their gradients will also be computed.
    """
    nclasses = y_batch.shape[1]

    x_batch.retain_grad()
    # placeholder for the jacobian
    jacobian = torch.zeros(x_batch.shape + (nclasses,),
                           dtype=x_batch.dtype, device=x_batch.device)
    # this mask tells torch to only compute the jacobian wrt. to the 0th index
    # of the output = ∇_x(f_0).
    indexing_mask = torch.zeros_like(y_batch)
    indexing_mask[:, 0] = 1.0

    for dim in range(nclasses):
        y_batch.backward(gradient=indexing_mask,
                         retain_graph=True, create_graph=create_graph)
        # fill the ith index with grad data of ∇_x(f_i).
        jacobian[..., dim] = x_batch.grad
        x_batch.grad.data.zero_()
        # shift the mask to compute the i+1th index of the output
        indexing_mask = torch.roll(indexing_mask, shifts=1, dims=1)

    # permute jacobian dimensions
    # from batch_size x input_dim x output_dim
    # to batch_size x output_dim x input_dim
    permute_dims = [0] + [len(jacobian.shape) - 1] + \
        list(range(1, len(jacobian.shape) - 1))

    return torch.permute(jacobian, permute_dims)
