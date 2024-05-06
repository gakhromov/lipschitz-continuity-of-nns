"""Training code for the models that computes the Lipschitz constant evolution 
during training.

Example of usage:
```bash
mkdir -p data/datasets
pipenv run python code/train.py --dataset MNIST1D --model_name FF_ReLU_256 \
    --loss CrossEntropy --optim SGD --lr 0.01 --lr_scheduler Const --seed 42 \
    --target_norm_grad 0.01 --compute_L 1 --compute_L_every 100 \
    --compute_L_for_first 100 --device CPU \
    --path ./my_experiment --dataset_path ./data 
```

"""
import argparse
import json
import logging
import logging.config
import random
from collections.abc import Callable
from pathlib import Path
from time import time
from typing import Any

import numpy as np
import torch
from adversarial_attacks import adverse_batch_pgd
from lipschitz import (compute_jac_norm,
                       compute_lipschitz_upper_bound_per_layer,
                       compute_upper_bound)
from sequential_models import Conv_Net, ReLU_Net, ResNet, SequentialModel
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.enums import Datasets, Devices, Losses, LR_Schedulers, Optims
from utils.functions import (build_run_name, get_grad_vector_of_params,
                             get_last_checkpointed_epoch, get_log_config,
                             get_results, get_scalar, get_tb_layouts,
                             get_vector_of_params, read_state_dict)
from vit_pytorch import ViT


def compute_lipschitz_bounds(
    model: SequentialModel,
    layers_to_look_at: list[int],
    train_dataloader: DataLoader,
    device: torch.device,
    ord: int | float = 2,
    verbose: bool = False,
) -> dict[str, tuple[float, float, float, float]]:
    """Computes the Lipschitz constant bounds for the Sequential model at 
    specified layers.

    Parameters
    ----------
    model
        Model object.
    layers_to_look_at
        List of layers to compute the Lipschitz constant for.
    train_dataloader
        Train dataloader.
    device
        Torch device to use for the computation.
    ord
        The order of the norm, by default 2. Possible options: 1, 2, torch.inf
    verbose, optional
        Log info on Lipschitz computation to the INFO log, by default False.

    Returns
    -------
        A dictionary, where each key is the string of the layer index and the 
        value is a tuple of 4 values: lower bound, mean norm, rms norm, upper 
        bound for the model at this layer.

    Raises
    ------
        NotImplemented error if order is not in [1, 2, torch.inf]
    """
    if ord not in [1, 2, torch.inf]:
        raise NotImplementedError

    ord_local = ord
    if model.dims[-1] == 1:
        # use the dual norm
        if ord == torch.inf:
            ord_local = 1
        if ord == 1:
            ord_local = torch.inf

    results = {}

    # supremum of the norms
    lower_bounds = dict([(str(l), torch.tensor(0.0).to(device))
                        for l in layers_to_look_at])
    # mean of the norms
    mean_bounds = dict([(str(l), torch.tensor(0.0).to(device))
                       for l in layers_to_look_at])
    # rms of the norms
    rms_bounds = dict([(str(l), torch.tensor(0.0).to(device))
                      for l in layers_to_look_at])

    n_samples = 0

    if verbose:
        logging.info("Computing the lower bound...")
    for i, (x_batch, y_batch) in enumerate(train_dataloader):
        if verbose:
            logging.info(f"Processing batch {i+1}/{len(train_dataloader)}...")

        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        n_samples += x_batch.shape[0]

        for layer in layers_to_look_at:
            def m(input): return model.forward_up_to_k_layer(input, layer)
            norms = compute_jac_norm(m, x_batch, ord=ord_local)
            # lower bound is the supremum over norms
            lower = torch.max(norms)
            if lower > lower_bounds[str(layer)]:
                lower_bounds[str(layer)] = lower

            mean_bounds[str(layer)] += torch.sum(norms)
            rms_bounds[str(layer)] += torch.sum(norms**2)

    if verbose:
        logging.info("Lower bound computed!")

    # compute this for the upper_bounds
    # NOTE: indexation of layers in per_layer_Lipschitzness works in the following way:
    # layer 0 == input (always 1-Lipschitz)
    # layer 1 == first layer applied (compute Lip. wrt. to input)
    # layer 2 == first two layers applied (compute Lip. wrt. to output of layer 1)
    # ...

    if verbose:
        logging.info("Computing the upper bound...")
    per_layer_Lipschitzness = [torch.tensor(1.0)]
    for i in range(len(model.layers)):
        if verbose:
            logging.info(f"Processing layer {i+1}/{len(model.layers)}...")
        per_layer_Lipschitzness.append(
            # collapse all nested layers to a singular layer
            compute_upper_bound(
                compute_lipschitz_upper_bound_per_layer(
                    model.layers[i], model.layer_input_shapes[i], ord=ord
                )
            )
        )
    per_layer_Lipschitzness = torch.Tensor(per_layer_Lipschitzness)

    if verbose:
        logging.info("Upper bound computed!")

    # format results
    for layer in layers_to_look_at:
        # upper bound is the product of per-layer Lipschitz constants)
        upper_bound = torch.prod(per_layer_Lipschitzness[: layer + 1])
        results[str(layer)] = (
            # sup of norms (lower bound)
            float(lower_bounds[str(layer)].item()),
            # mean of norms (mean bound)
            float((mean_bounds[str(layer)] / n_samples).item()),
            # rms of norms (rms bound)
            float(torch.sqrt(rms_bounds[str(layer)] / n_samples).item()),
            # upper bound
            float(upper_bound.item()),
        )

    return results


def run_epoch(
    epoch_number: int,
    model: torch.nn.Module,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    loss_f: Callable,
    optimiser: torch.optim.Optimizer,
    lr_scheduler: Any,
    device: torch.device,
    layers_to_look_at: list,
    writer: SummaryWriter,
    batch_counter: int,
    min_num_epochs: int,
    max_num_epochs: int,
    theta_0: float | None = None,
    compute_lip_this_epoch: bool = True,
    adv_attack: Callable | None = None,
) -> tuple[float, float, float, float, float, torch.Tensor, int]:
    """Run one epoch of the model.
    This includes training, testing, computing the Lipschitz constant and 
    logging epoch details. Note that model updates do not happen for the 0th 
    epoch.

    Parameters
    ----------
    epoch_number
        Epoch number. If `epoch_index=0`, training is not performed (only stats 
        are computed). Otherwise used for logging.
    model
        Model object.
    train_dataloader
        Train Dataloader.
    test_dataloader
        Test Dataloader.
    loss_f
        Loss function with reduction="sum".
    optimiser
        Optimiser object.
    lr_scheduler
        LR Scheduler object.
    device
        Torch device.
    layers_to_look_at
        List of layers to compute the Lipschitz constant for.
        By defualt contains one element with the last layer.
    writer
        Tensorboard Writer.
    batch_counter
        Global counter for the number of batches trained for.
    min_num_epochs
        Minimum number of epochs that the model will train for. Used for 
        logging.
    max_num_epochs
        Maximum number of epochs that the model is allowed to train for. Used 
        for logging.
    theta_0, optional
        Parameter vector at initialisation, by default None. Not required for 
        the 0th epoch.
    compute_lip_this_epoch
        Flag that controls whether to compute the Lipschitz constant bounds this
        epoch or not. By default True.
    adv_attack
        Function that adversarially transforms `x_batch`. This function should 
        take `x_batch, y_batch, model` as input and should output `x_adv_batch`.
        Metrics are calculated for adversed data. Lipschitz constant lower bound
        is computed for clean samples.

    Returns
    -------
        Return the following end-of-epoch stats, computed for the model after 
        the last update: mean `train_loss`, mean `train_accuracy`, mean 
        `test_loss`, mean `test_accuracy`, `norm_grad_of_params_this_epoch`, 
        `theta_t` parameter vector and the `batch_counter`.
    """
    epoch_start_time = time()

    # do not train at 0, this should only evaluate stuff for the init setting
    if epoch_number != 0:
        # train
        model.train()
        batch_counter = train_model_epoch(
            model,
            train_dataloader,
            loss_f,
            optimiser,
            lr_scheduler,
            device,
            batch_counter,
            adv_attack=adv_attack,
        )

    # get metrics
    model.train()
    train_loss, train_accuracy, grad_of_params_this_epoch = \
        compute_loss_and_grad_vector_of_params(
            model, train_dataloader, loss_f, device, adv_attack=None
        )
    model.eval()
    test_loss, test_accuracy = test_model_epoch(
        model, test_dataloader, loss_f, device, adv_attack=None)

    # get norms of gradients for each parameter
    norm_grad_of_params_this_epoch = torch.linalg.norm(
        grad_of_params_this_epoch, 2).item()

    theta_t = get_vector_of_params(model)

    # compute the lipschitz constant
    lipschitz_bounds = None

    if compute_lip_this_epoch:
        lipschitz_bounds = compute_lipschitz_bounds(
            model, layers_to_look_at, train_dataloader, device
        )

    # log everything
    log_epoch_stats(
        writer,
        epoch_number,
        train_loss,
        test_loss,
        train_accuracy,
        test_accuracy,
        norm_grad_of_params_this_epoch,
        min_num_epochs,
        max_num_epochs,
        epoch_start_time,
        layers_to_look_at,
        lr_scheduler,
        theta_t,
        theta_0,
        lipschitz_bounds,
    )

    return (
        train_loss,
        train_accuracy,
        test_loss,
        test_accuracy,
        norm_grad_of_params_this_epoch,
        theta_t,
        batch_counter,
    )


def log_epoch_stats(
    writer: SummaryWriter,
    epoch_number: int,
    train_loss: float,
    test_loss: float,
    train_accuracy: float,
    test_accuracy: float,
    norm_grad_of_params_this_epoch: float,
    min_num_epochs: int,
    max_num_epochs: int,
    epoch_start_time: float,
    layers_to_look_at: list,
    lr_scheduler: Any,
    theta_t: torch.Tensor,
    theta_0: torch.Tensor | None = None,
    lipschitz_bounds: dict | None = None,
):
    # log to tensorboard
    writer.add_scalar("norm_grad_of_params",
                      norm_grad_of_params_this_epoch, epoch_number)
    writer.add_scalar("params/norm_theta_t",
                      torch.linalg.norm(theta_t, 2).item(), epoch_number)
    if epoch_number == 0:
        writer.add_scalar("params/norm_dtheta_t0", 0, epoch_number)
    else:
        writer.add_scalar(
            "params/norm_dtheta_t0", torch.linalg.norm(
                theta_t - theta_0, 2).item(), epoch_number
        )

    writer.add_scalar("loss/train", train_loss, epoch_number)
    writer.add_scalar("accuracy/train", train_accuracy, epoch_number)
    writer.add_scalar("loss/test", test_loss, epoch_number)
    writer.add_scalar("accuracy/test", test_accuracy, epoch_number)

    if lipschitz_bounds is not None:
        for layer in layers_to_look_at:
            lower, mean, rms, upper = lipschitz_bounds[str(layer)]
            writer.add_scalar(f"L_lower/layer_{layer}", lower, epoch_number)
            writer.add_scalar(f"L_mean/layer_{layer}", mean, epoch_number)
            writer.add_scalar(f"L_rms/layer_{layer}", rms, epoch_number)
            writer.add_scalar(f"L_upper/layer_{layer}", upper, epoch_number)

    # log to the log file
    if epoch_number <= min_num_epochs:
        extension = ""
    else:
        extension = f" (training extended to {max_num_epochs} epochs)"

    logging.info(f"Epoch = {epoch_number}/{min_num_epochs}{extension}")
    logging.info(f"Gradient norm = {norm_grad_of_params_this_epoch}, LR at the "
                 f"end of the epoch = {lr_scheduler.get_last_lr()[0]}")
    logging.info("")
    logging.info("Train/test metrics:")
    logging.info(f"Train loss = {train_loss}, train accuracy = "
                 f"{train_accuracy*100}%")
    logging.info(f"Test loss = {test_loss}, test accuracy = "
                 f"{test_accuracy*100}%")

    if lipschitz_bounds is not None:
        logging.info("")
        logging.info(f"Lipschitz bounds:")

        for layer in layers_to_look_at:
            lower, mean, rms, upper = lipschitz_bounds[str(layer)]
            logging.info(f"Lower Lipschitz @layer_{layer} = {lower}")
            logging.info(f"Mean Lipschitz @layer_{layer} = {mean}")
            logging.info(f"RMS Lipschitz @layer_{layer} = {rms}")
            logging.info(f"Upper Lipschitz @layer_{layer} = {upper}")

    logging.info("")
    time_elapsed = time() - epoch_start_time
    logging.info(f"Time for this epoch = {time_elapsed}")
    logging.info("-" * 50)


def train_model_epoch(
    model: SequentialModel,
    train_dataloader: DataLoader,
    loss_f: Callable,
    optimiser: torch.optim.Optimizer,
    lr_scheduler: Any,
    device: torch.device,
    batch_counter: int,
    adv_attack: Callable = None,
) -> int:
    for x_batch, y_batch in train_dataloader:
        curr_batch_size = x_batch.shape[0]
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        if adv_attack is not None:
            # adversarial training
            x_adv_batch = adv_attack(x_batch, y_batch, model)
            output = model(x_adv_batch)
        else:
            output = model(x_batch)
        # compute loss
        loss = loss_f(output, y_batch) / curr_batch_size

        # train model
        optimiser.zero_grad()
        loss.backward()

        # gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 50.0)

        # step the optimiser, lr scheduler
        optimiser.step()
        lr_scheduler.step()

        # increase batch counter
        batch_counter += 1
    return batch_counter


def compute_loss_and_grad_vector_of_params(
    model: SequentialModel,
    dataloader: DataLoader,
    loss_f: Callable,
    device: torch.device,
    adv_attack: Callable = None,
) -> tuple[float, float, torch.Tensor]:
    loss = 0
    n_correct = 0
    n_samples = 0

    # ∇_W L(W,X) is a flattened vector of all parameters ∈ R^n_params
    grad_vec = None

    for x_batch, y_batch in dataloader:
        model.zero_grad()

        curr_batch_size = x_batch.shape[0]
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        if adv_attack is not None:
            # adversarial training
            x_adv_batch = adv_attack(x_batch, y_batch, model)
            output = model(x_adv_batch)
        else:
            output = model(x_batch)
        # compute loss
        curr_loss = loss_f(output, y_batch)
        loss += curr_loss

        # compute gradient vector of parameters
        curr_loss.backward()
        curr_grad_vec = get_grad_vector_of_params(model)
        grad_vec = curr_grad_vec if grad_vec is None else grad_vec + \
            curr_grad_vec

        # compute accuracy
        # get class number
        prediction = torch.argmax(output, 1)
        if len(y_batch.shape) > 1:
            # if labels are one-hot-encoded, transform to class numbers
            y_batch = torch.argmax(y_batch, 1)
        n_correct += (prediction == y_batch).sum().item()
        n_samples += curr_batch_size

    # normalise the loss and accuracy
    accuracy = n_correct / n_samples
    loss = loss / n_samples
    grad_vec = grad_vec / n_samples
    return loss.item(), accuracy, grad_vec


def test_model_epoch(
    model: SequentialModel,
    dataloader: DataLoader,
    loss_f: Callable,
    device: torch.device,
    adv_attack: Callable | None = None,
) -> tuple[float, float]:
    loss = 0
    n_correct = 0
    n_samples = 0

    for x_batch, y_batch in dataloader:
        curr_batch_size = x_batch.shape[0]
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        if adv_attack is not None:
            # adversarial training
            x_adv_batch = adv_attack(x_batch, y_batch, model)
            output = model(x_adv_batch)
        else:
            output = model(x_batch)

        with torch.no_grad():
            curr_loss = loss_f(output, y_batch)
        loss += curr_loss

        # compute accuracy
        # get class number
        prediction = torch.argmax(output, 1)
        if len(y_batch.shape) > 1:
            # if labels are one-hot-encoded
            # if labels are one-hot-encoded, transform to class numbers
            y_batch = torch.argmax(y_batch, 1)
        n_correct += (prediction == y_batch).sum().item()
        n_samples += curr_batch_size

    accuracy = n_correct / n_samples
    loss = loss / n_samples
    return loss.item(), accuracy


def get_model(model_name: str,
              dims: list,
              seed=42,
              **kwargs) -> SequentialModel:
    """Get model object from its name.

    Parameters
    ----------
    model_name
        Name of the model with the width specification.
    dims
        Dimensions of the input and the output in the form: 
        [input_shape, output_shape].
    seed, optional
        Random seed used to initialise the model, by default 42.

    Returns
    -------
        Model that is derived from SequentialModel class.

    Raises
    ------
    NameError
        This exception is raised when the model name was not found.
    """
    # fix the seed for model initialisation
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if model_name.startswith("FF_ReLU"):
        widths = list(map(int, model_name.split("_")[2:]))
        return ReLU_Net(widths, dims, bias=False)

    if model_name.startswith("Bias_FF_ReLU"):
        widths = list(map(int, model_name.split("_")[3:]))
        return ReLU_Net(widths, dims, bias=True)

    if model_name.startswith("LeakyOutput_FF_ReLU"):
        widths = list(map(int, model_name.split("_")[3:]))
        return ReLU_Net(widths, dims, leaky_output=True, bias=False)

    if model_name.startswith("Bias_LeakyOutput_FF_ReLU"):
        widths = list(map(int, model_name.split("_")[4:]))
        return ReLU_Net(widths, dims, leaky_output=True, bias=True)

    if model_name.startswith("CNN_"):
        width = int(model_name.split("_")[1])
        return Conv_Net(width, dims, bias=False)

    if model_name.startswith("GELU_CNN_"):
        width = int(model_name.split("_")[2])
        return Conv_Net(width, dims, bias=False, activation=torch.nn.GELU())

    if model_name.startswith("ResNet_"):
        width = int(model_name.split("_")[1])
        return ResNet(width, dims, **kwargs)

    if model_name == "ViT_Mini":
        # 22 682 params for CIFAR-10
        return ViT(
            image_size=dims[0][-1],
            patch_size=8,
            num_classes=dims[-1],
            depth=2,
            heads=2,
            mlp_dim=32,
            dim=16,
            dropout=0.1,
            emb_dropout=0.1,
        )

    if model_name == "ViT_Interp":
        # 49 066 params for CIFAR-10
        return ViT(
            image_size=dims[0][-1],
            patch_size=8,
            num_classes=dims[-1],
            depth=2,
            heads=2,
            mlp_dim=64,
            dim=32,
            dropout=0.1,
            emb_dropout=0.1,
        )

    if model_name == "ViT_Interp2":
        # 97 610 params for CIFAR-10
        return ViT(
            image_size=dims[0][-1],
            patch_size=8,
            num_classes=dims[-1],
            depth=2,
            heads=2,
            mlp_dim=64,
            dim=64,
            dropout=0.1,
            emb_dropout=0.1,
        )

    if model_name == "ViT_Medium":
        # 35 608 714 params for CIFAR-10
        return ViT(
            image_size=dims[0][-1],
            patch_size=8,
            num_classes=dims[-1],
            depth=6,
            heads=6,
            mlp_dim=3072,
            dim=768,
            dropout=0.1,
            emb_dropout=0.1,
        )

    if model_name == "ViT_Base":
        # 85 200 010 for CIFAR-10
        return ViT(
            image_size=dims[0][-1],
            patch_size=8,
            num_classes=dims[-1],
            depth=12,
            heads=12,
            mlp_dim=3072,
            dim=768,
            dropout=0.1,
            emb_dropout=0.1,
        )

    if model_name.startswith("ViT_"):
        width = int(model_name.split("_")[1])
        return ViT(
            image_size=dims[0][-1],
            patch_size=8,
            num_classes=dims[-1],
            depth=6,
            heads=6,
            mlp_dim=4 * width,
            dim=width,
            dropout=0.0,
            emb_dropout=0.0,
        )

    logging.error("Model does not exist")
    raise NameError(name="model_name")


def initiate_model_training(
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    dataset_dims: list[int | list],
    model_name: str,
    loss_f: Callable,
    optim: Optims,
    lr: float,
    lr_scheduler_enum: LR_Schedulers,
    target_norm_grad: float,
    path: Path,
    dataset_path: Path,
    seed: int,
    device: torch.device,
    min_num_epochs: int,
    max_num_epochs: int,
    save_model_every: int,
    compute_L: int,
    compute_L_every: int,
    compute_L_for_first: int,
    dataset_noise: float,
    batch_size: int,
    batch_size_test: int,
    alpha_shuffle: float,
    data_num_workers: int,
    load_model: bool,
    load_from_epoch: int,
    runtimestamp: int,
    run_name: str,
    adv_attack: Callable = None,
):
    # setup random seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # check if we need ohe encoding
    if isinstance(loss_f, torch.nn.MSELoss):
        one_hot_encode_y = True
    else:
        one_hot_encode_y = False

    # init model
    model = get_model(model_name, dataset_dims, seed=seed).to(device)

    # prepare for training
    optimiser = optim.value(model.parameters(), lr=lr)

    # lr decay
    lr_scheduler = lr_scheduler_enum.value(optimiser,
                                           batch_size,
                                           len(train_dataloader.dataset))

    # load model to continue training
    if load_model:
        # load model at init to get theta_0
        state_dict = read_state_dict(run_name, path, device, epoch=0)
        model.load_state_dict(state_dict["model"])
        theta_0 = get_vector_of_params(model)

        # get exact epoch number
        if load_from_epoch == -1:
            load_from_epoch = get_last_checkpointed_epoch(run_name, path)
        else:
            load_from_epoch = load_from_epoch
        # load model and scheduler
        logging.info(f"Continue training from epoch {load_from_epoch}...")
        state_dict = read_state_dict(run_name,
                                     path,
                                     device,
                                     epoch=load_from_epoch)
        model.load_state_dict(state_dict["model"])
        model = model.to(device)
        # step the scheduler
        [lr_scheduler.step() for _ in range(load_from_epoch)]

        # get the grad norm from previous epoch
        norm_grad_of_params_this_epoch = get_scalar(
            get_results(path), "norm_grad_of_params")[1][-1]

    # check that folder for checkpoints exists
    (path / "checkpoints" / run_name).mkdir(exist_ok=True, parents=True)

    # setup layers to record lipschitz
    num_layers = len(model.layers)

    # NOTE: indexation of layers works in the following way:
    # layer 0 == input
    # layer 1 == first layer applied
    # layer 2 == first two layers applied
    # ...

    # Look at the most important layers for FF_ReLU
    # skip 0-th layer (just input), skip linear layers without activation
    # applied
    # layers_to_look_at = list(range(num_layers+1))[2::2]

    # Look only at the last layer
    layers_to_look_at = [num_layers]

    # logging
    writer = SummaryWriter(log_dir=str(
        (path / "runs" / run_name).resolve()))
    writer.add_custom_scalars(get_tb_layouts(layers_to_look_at))

    if not load_model:
        # run 0th epoch
        (
            train_loss,
            train_accuracy,
            test_loss,
            test_accuracy,
            norm_grad_of_params_this_epoch,
            theta_0,
            _,
        ) = run_epoch(
            0,
            model,
            train_dataloader,
            test_dataloader,
            loss_f,
            optimiser,
            lr_scheduler,
            device,
            layers_to_look_at,
            writer,
            0,
            min_num_epochs,
            max_num_epochs,
            compute_lip_this_epoch=(compute_L_for_first != 0) and compute_L,
            adv_attack=adv_attack,
        )
        # save model and scheduler
        torch.save(
            {
                "model": model.state_dict(),
                "scheduler": lr_scheduler.state_dict(),
            },
            path / "checkpoints" / run_name / "model_on_epoch_0",
        )

    starting_epoch = 1
    if load_model:
        starting_epoch = load_from_epoch + 1

    # main training loop
    # set epoch = 0 in case we train for 0 epochs (this variable is used later
    # for final logs)
    epoch = 0
    batch_counter = 0

    for epoch in range(starting_epoch, max_num_epochs + 1):
        # determine whether to compute Lipschitz this epoch
        compute_lip_this_epoch = (
            (epoch % compute_L_every == 0)
            or (epoch <= compute_L_for_first)
            or (epoch == min_num_epochs)
            or (epoch == max_num_epochs)
            or (
                (epoch > min_num_epochs)
                # stopping criteria
                and (norm_grad_of_params_this_epoch <= target_norm_grad)
            )
        ) and compute_L

        # run epoch
        (
            train_loss,
            train_accuracy,
            test_loss,
            test_accuracy,
            norm_grad_of_params_this_epoch,
            _,
            batch_counter,
        ) = run_epoch(
            epoch,
            model,
            train_dataloader,
            test_dataloader,
            loss_f,
            optimiser,
            lr_scheduler,
            device,
            layers_to_look_at,
            writer,
            batch_counter,
            min_num_epochs,
            max_num_epochs,
            theta_0,
            compute_lip_this_epoch=compute_lip_this_epoch,
            adv_attack=adv_attack,
        )

        # save model
        if (
            (epoch % save_model_every == 0)
            or (epoch == min_num_epochs)
            or (epoch == max_num_epochs)
            or (
                (epoch > min_num_epochs)
                # stopping criteria
                and (norm_grad_of_params_this_epoch <= target_norm_grad)
            )
        ):  # save for last epoch before stop
            # save model and scheduler
            torch.save(
                {
                    "model": model.state_dict(),
                    "scheduler": lr_scheduler.state_dict(),
                },
                path / "checkpoints" / run_name /
                f"model_on_epoch_{epoch}",
            )

        # stop the training loop
        if (norm_grad_of_params_this_epoch <= target_norm_grad) and (
            epoch >= min_num_epochs
        ):
            break

    # final metrics for Lip
    final_metrics = {
        "hparam/train_loss": train_loss,
        "hparam/train_accuracy": train_accuracy,
        "hparam/test_loss": test_loss,
        "hparam/test_accuracy": test_accuracy,
        "hparam/last_epoch": epoch,
        "hparam/last_batch": batch_counter,
    }

    hparams = {
        "lr": lr,
        "optim": optim.name,
        "loss": loss_f.__class__.__name__,
        "lr_scheduler": lr_scheduler_enum.name,
        "target_norm_grad": target_norm_grad,
        "seed": seed,
        "model_name": model_name,
        "min_num_epochs": min_num_epochs,
        "max_num_epochs": max_num_epochs,
        "save_model_every": save_model_every,
        "compute_L": compute_L,
        "compute_L_every": compute_L_every,
        "compute_L_for_first": compute_L_for_first,
        "dataset_noise": dataset_noise,
        "batch_size": batch_size,
        "batch_size_test": batch_size_test,
        "alpha_shuffle": alpha_shuffle,
        "data_num_workers": data_num_workers,
        "load_model": load_model,
        "load_from_epoch": load_from_epoch,
        "runtimestamp": runtimestamp,
    }

    writer.add_hparams(hparams, final_metrics)
    writer.flush()
    writer.close()

    # run get results to generate a csv file from TensorBoard logs
    logging.info("Parsing TensorBoard logs...")
    get_results(path / "runs" / run_name, regen_csv=True)

    logging.info("Training finished successfully")
    return model


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", dest="dataset", type=str)
    parser.add_argument("--dataset_noise",
                        dest="dataset_noise", default=0.0, type=float)
    parser.add_argument("--batch_size", dest="batch_size",
                        default=512, type=int)
    parser.add_argument("--batch_size_test",
                        dest="batch_size_test", default=512, type=int)
    parser.add_argument("--data_num_workers",
                        dest="data_num_workers", default=0, type=int)

    parser.add_argument("--min_num_epochs",
                        dest="min_num_epochs", default=10_000, type=int)
    parser.add_argument("--max_num_epochs",
                        dest="max_num_epochs", default=1_000_000, type=int)

    # save model every X epoch
    parser.add_argument("--save_model_every",
                        dest="save_model_every", default=100, type=int)
    # compute Lipschitz every X epoch
    parser.add_argument("--compute_L", dest="compute_L",
                        default=1, type=int, choices=[0, 1])
    parser.add_argument("--compute_L_every",
                        dest="compute_L_every", default=100, type=int)
    # compute Lipschitz for the first X epochs
    parser.add_argument("--compute_L_for_first",
                        dest="compute_L_for_first", default=1000, type=int)

    parser.add_argument("--model_name", dest="model_name", type=str)
    parser.add_argument("--loss", dest="loss", type=str)
    parser.add_argument("--optim", dest="optim", type=str)
    parser.add_argument("--lr", dest="lr", type=float)
    parser.add_argument("--lr_scheduler", dest="lr_scheduler", type=str)
    parser.add_argument("--alpha_shuffle",
                        dest="alpha_shuffle", default=0.0, type=float)
    parser.add_argument("--target_norm_grad",
                        dest="target_norm_grad", type=float)

    parser.add_argument("--adv_train", dest="adv_train",
                        type=str, default=None)

    parser.add_argument("--seed", dest="seed", type=int)

    parser.add_argument("--load_model", dest="load_model",
                        default=0, choices=[0, 1], type=int)
    parser.add_argument("--load_from_epoch",
                        dest="load_from_epoch", default=-1, type=int)
    parser.add_argument("--runtimestamp", dest="runtimestamp",
                        default=int(time()), type=int)

    parser.add_argument("--device", dest="device", type=str, default="CPU")

    parser.add_argument("--dataset_path", dest="dataset_path", type=str)
    parser.add_argument("--path", dest="path", type=str)

    args = parser.parse_args()

    # form run name
    run_name = build_run_name(
        args.dataset,
        args.model_name,
        args.loss,
        args.optim,
        args.lr,
        args.lr_scheduler,
        args.alpha_shuffle,
        args.seed,
        args.target_norm_grad,
        args.runtimestamp,
    )
    path = Path(args.path)
    dataset_path = Path(args.dataset_path)

    # save arguments
    path_to_args = path / "args"
    path_to_args.mkdir(parents=True, exist_ok=True)

    arg_file = path_to_args / f"{run_name}.json"
    if arg_file.exists():
        # dump arguments in a separate key to store old args
        with arg_file.open(mode="r") as f:
            arg_dict = json.load(f)

        if "new_args" not in arg_dict.keys():
            arg_dict["new_args"] = []
        arg_dict["new_args"].append(vars(args))

        with arg_file.open(mode="w") as f:
            json.dump(arg_dict, f)

    else:
        # dump arguments
        with arg_file.open(mode="w") as f:
            json.dump(vars(args), f)

    # setup logging
    logging.config.dictConfig(get_log_config(path, run_name))

    # make some checks
    if args.dataset not in Datasets.__members__:
        logging.error(f"Dataset {args.dataset} is unknown. Known datasets: "
                      f"{Datasets.__members__}")
        exit(1)

    if args.optim not in Optims.__members__:
        logging.error(f"Optimiser {args.optim} is unknown. Known optimisers: "
                      f"{Optims.__members__}")
        exit(1)

    if args.loss not in Losses.__members__:
        logging.error(f"Loss {args.loss} is unknown. Known losses: "
                      f"{Losses.__members__}")
        exit(1)

    if args.min_num_epochs > args.max_num_epochs:
        logging.error(f"min_num_epochs {args.min_num_epochs} should be <= than "
                      f"max_num_epochs {args.max_num_epochs}")
        exit(1)

    adv_attack = None
    if args.adv_train is not None:
        if args.adv_train.startswith("PGD_L2_CE_") or \
                args.adv_train.startswith("PGD_Linf_CE_"):
            if "L2" in args.adv_train:
                ord = 2
            if "Linf" in args.adv_train:
                ord = torch.inf

            epsilon = float(args.adv_train.split("_")[3])

            def adv_attack(x_batch, y_batch, model):
                return adverse_batch_pgd(
                    x_batch,
                    y_batch,
                    model,
                    ord,
                    epsilon,
                    loss_f=torch.nn.CrossEntropyLoss(reduction="mean"),
                    max_steps=10,
                    gamma=20,
                    seed=42,
                )

    logging.info(f"Starting run {run_name}")
    logging.info("Run params:")
    logging.info(json.dumps(vars(args)))
    logging.info("-" * 50)

    # check if we need ohe encoding
    if args.loss == Losses.MSE.name:
        one_hot_encode_y = True
    else:
        one_hot_encode_y = False

    # load dataset
    dataset = Datasets[args.dataset]
    (
        train_dataset,
        test_dataset,
        train_dataloader,
        test_dataloader,
        dataset_dims,
    ) = dataset.value(
        batch_size=args.batch_size,
        batch_size_test=args.batch_size_test,
        noise_scale=args.dataset_noise,
        one_hot_encode_y=one_hot_encode_y,
        alpha_shuffle=args.alpha_shuffle,
        dataset_path=dataset_path,
        num_workers=args.data_num_workers,
    )

    device = Devices[args.device].value
    loss_f = Losses[args.loss].value()
    optim = Optims[args.optim]
    lr_scheduler = LR_Schedulers[args.lr_scheduler]

    # start training
    initiate_model_training(
        train_dataloader,
        test_dataloader,
        dataset_dims,
        args.model_name,
        loss_f,
        optim,
        args.lr,
        lr_scheduler,
        args.target_norm_grad,
        path,
        dataset_path,
        args.seed,
        device,
        args.min_num_epochs,
        args.max_num_epochs,
        args.save_model_every,
        args.compute_L,
        args.compute_L_every,
        args.compute_L_for_first,
        args.dataset_noise,
        args.batch_size,
        args.batch_size_test,
        args.alpha_shuffle,
        args.data_num_workers,
        args.load_model,
        args.load_from_epoch,
        args.runtimestamp,
        run_name,
        adv_attack=adv_attack,
    )
