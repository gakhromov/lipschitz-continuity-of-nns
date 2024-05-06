"""Miscellaneous utility functions for the project."""
import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tbparse import SummaryReader


def seed_worker(worker_id):
    """This function is used to seed the workers in the dataloader."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def check_basic_block_structure(layer: torch.nn.Module):
    basic_block_struc = [
        "",
        "conv1",
        "bn1",
        "relu",
        "conv2",
        "bn2",
        "downsample",
        "downsample.0",
        "downsample.1",
    ]
    layer_struc = [i for i, _ in layer.named_modules()]

    # downsample layers may not be present
    return (layer_struc == basic_block_struc) or \
        (layer_struc == basic_block_struc[:-3])


def check_bottleneck_structure(layer: torch.nn.Module):
    bottleneck_struc = [
        "",
        "conv1",
        "bn1",
        "conv2",
        "bn2",
        "conv3",
        "bn3",
        "relu",
        "downsample",
        "downsample.0",
        "downsample.1",
    ]
    layer_struc = [i for i, _ in layer.named_modules()]

    # downsample layers may not be present
    return (layer_struc == bottleneck_struc) or \
        (layer_struc == bottleneck_struc[:-3])


def get_vector_of_params(model: torch.nn.Module) -> torch.Tensor:
    with torch.no_grad():
        return torch.hstack([
            layer.flatten().detach() for layer in model.parameters()])


def get_grad_vector_of_params(model: torch.nn.Module) -> torch.Tensor:
    """Computes the flattened vector of the gradient of the loss wrt. to 
    parameters.

    Note
    ----
        Requires a loss backward pass before the call.

    Parameters
    ----------
    model
        Model object.

    Returns
    -------
        The flattened vector of parameter gradients.
    """
    # get a list of norms of gradients of each param
    return torch.hstack([layer.grad.flatten() for layer in model.parameters()])


def build_run_name(
    dataset: str,
    model_name: str,
    loss: str,
    optim: str,
    lr: float,
    lr_scheduler: str,
    alpha_shuffle: float,
    seed: int,
    target_norm_grad: float,
    runtimestamp: int,
    regularisation: str = None,
) -> str:
    reg_str = ""
    if regularisation is not None:
        reg_str = "_" + regularisation
    return (
        f"{dataset}+{model_name}+{loss}+"
        f"{optim}_{lr}_{lr_scheduler}{reg_str}+alpha_{alpha_shuffle}+"
        f"seed_{seed}+target_{target_norm_grad}+"
        f"runtimestamp_{runtimestamp}"
    )


def get_tb_layouts(layers_to_look_at: list) -> dict:
    model_overview = {}
    for layer in layers_to_look_at:
        model_overview[f"Lipschitz @layer {layer}"] = [
            "Multiline",
            [f"L_lower/layer_{layer}", f"L_upper/layer_{layer}"],
        ]
    model_overview["Train and test loss"] = [
        "Multiline", ["loss/train", "loss/test"]]
    model_overview["Train loss and ||∇ Loss(θ)||"] = [
        "Multiline",
        ["loss/train", "norm_grad_of_params"],
    ]
    model_overview["Train and test accuracy"] = [
        "Multiline", ["accuracy/train", "accuracy/test"]]

    return {
        "Model overview": model_overview,
    }


def get_results(path: Path, regen_csv: bool = False) -> pd.DataFrame:
    """Parses the TensorBoard files to create a DataFrame.

    Parameters
    ----------
    path
        Path to the TensorBoard run logs.

    Returns
    -------
        Pandas DataFrame with all logged values.
    """
    # reading from csv is faster, thus save to csv
    if (path / "scalars.csv").exists() and not regen_csv:
        return pd.read_csv(path / "scalars.csv")

    # else get data and transform to csv
    results: pd.DataFrame = SummaryReader(path).scalars
    results.to_csv(path / "scalars.csv", index=False)
    return results


def get_scalar(results: pd.DataFrame, scalar: str) -> tuple[np.array, np.array]:
    """Returns the value and steps at which it was computed from the TensorBoard 
    DataFrame.

    Parameters
    ----------
    results
        TensorBoard DataFrame.
    scalar
        The name of the scalar.

    Returns
    -------
        Numpy array of steps and a numpy array of values.
    """
    filtered = results[results.tag == scalar]
    return np.array(filtered.step), np.array(filtered.value)


def read_state_dict(run_full_name: str, path: Path, device: torch.device,
                    epoch: int = -1):
    path_to_model_checkpoints = path / "checkpoints" / run_full_name
    # check for checkpoints
    if not path_to_model_checkpoints.exists():
        logging.error(f"Error: Checkpoints for model {run_full_name} do not "
                      "exist, impossible to load the model")
        raise NameError(name=run_full_name)

    epoch_to_load = epoch
    if epoch == -1:
        epoch_to_load = get_last_checkpointed_epoch(run_full_name, path)

    state_dict = torch.load(
        path_to_model_checkpoints / f"model_on_epoch_{epoch_to_load}",
        map_location=device
    )
    return state_dict


def get_last_checkpointed_epoch(run_full_name: str, path: Path) -> int:
    path_to_model_checkpoints = path / "checkpoints" / run_full_name
    checkpoints = [i.name for i in path_to_model_checkpoints.iterdir()]
    epochs_saved = [int(x.split("_")[-1]) for x in checkpoints]
    return max(epochs_saved)


def get_log_config(path: Path, run_name: str) -> dict:
    path_to_logs = path / "logs"
    path_to_logs.mkdir(parents=True, exist_ok=True)

    return {
        "version": 1,
        "disable_existing_loggers": True,
        "formatters": {
            "standard": {
                "format": "%(asctime)s %(levelname)s %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "handlers": {
            # print to file
            "file": {
                "level": "INFO",
                "formatter": "standard",
                "class": "logging.FileHandler",
                "filename": str((path_to_logs / f"{run_name}.log").resolve()),
                "mode": "a",
            },
            # print to console
            "console": {
                "level": "INFO",
                "formatter": "standard",
                "class": "logging.StreamHandler",
            },
        },
        "loggers": {
            "": {  # root logger
                "handlers": ["file", "console"],
                "level": "INFO",
                "propagate": False,
            }
        },
    }
