"""MNIST dataset wrapper."""
import logging
from pathlib import Path

import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from datasets.base_dataset import BaseDataset
from torch.utils.data import DataLoader
from utils.functions import seed_worker


class MNISTWrapper(BaseDataset):
    def __init__(
        self,
        dataset_path: Path,
        is_train: bool,
        noise_scale: float,
        one_hot_y: bool,
        dataset_len: int,
        alpha_shuffle: float = 0.0,
        flatten: bool = False,
        size: int = 32,
        **kwargs,
    ):
        super(MNISTWrapper, self).__init__(
            dataset_path=dataset_path,
            is_train=is_train,
            noise_scale=noise_scale,
            one_hot_y=one_hot_y,
            dataset_len=dataset_len,
            alpha_shuffle=alpha_shuffle,
            flatten=flatten,
            size=size,
            **kwargs,
        )

    def bake_dataset(
        self,
        dataset_path: Path,
        is_train: bool,
        noise_scale: float,
        one_hot_y: bool,
        dataset_len: int,
        alpha_shuffle: float = 0.0,
        flatten: bool = False,
        size: int = 28,
    ):
        np.random.seed(42)
        torch.manual_seed(42)

        full_ds_path = (
            dataset_path
            / "datasets"
            / (f"mnist_{is_train}_{flatten}_{size}_{one_hot_y}_{dataset_len}_"
               f"{alpha_shuffle}.pt")
        )

        # check that this dataset exists
        if full_ds_path.exists():
            baked_x, baked_y = torch.load(
                full_ds_path,
                map_location=torch.device("cpu"),
            )
            return baked_x, baked_y

        # if not, create the dataset
        transforms = [T.ToTensor()]

        # add gaussian noise
        if noise_scale > 0.0:

            def add_noise(x: torch.Tensor) -> torch.Tensor:
                noise = (
                    torch.normal(torch.zeros_like(
                        x), torch.ones_like(x) * noise_scale)
                    .type(x.dtype)
                    .to(x.device)
                )

                return x + noise

            transforms.append(T.Lambda(add_noise))

        transforms.append(T.Resize(size))
        if flatten:
            transforms.append(T.Lambda(torch.flatten))

        # download the dataset
        dataset = torchvision.datasets.MNIST(
            str((dataset_path / "datasets").resolve()),
            train=is_train,
            transform=T.Compose(transforms),
            target_transform=self.onehot if one_hot_y else None,
            download=True,
        )

        # sample subset
        subset_indices = np.random.choice(
            len(dataset), size=dataset_len, replace=False)

        # shuffle
        labels_map = subset_indices.copy()
        if alpha_shuffle > 0.0:
            labels_map = self.incremental_shuffle(labels_map, alpha_shuffle)

        # since all transforms are evaluated on each call,
        # we precompute the transforms for faster later computations.
        dataset_type = "train" if is_train else "test"
        logging.info(f"Start baking the {dataset_type} dataset... (pre-applying"
                     " all transformations and then saving the dataset)")

        x_arr = []
        y_arr = []
        for i in range(dataset_len):
            x, _ = dataset[subset_indices[i]]
            _, y = dataset[labels_map[i]]
            x_arr.append(x)
            y_arr.append(y)
        logging.info("Baking finished!")

        # in case we use OHE, we have a list of tensors in y_arr
        if one_hot_y:
            baked_x, baked_y = (torch.stack(x_arr), torch.stack(
                y_arr).type(torch.FloatTensor))
        # otherwise, we have an array of numbers
        else:
            baked_x, baked_y = (torch.stack(x_arr), torch.Tensor(
                y_arr).type(torch.LongTensor))

        # save:
        torch.save((baked_x, baked_y), full_ds_path)

        return baked_x, baked_y


def load_datasets(
    dataset_path: Path,
    batch_size: int = 128,
    batch_size_test: int = 128,
    noise_scale: float = 0.0,
    one_hot_encode_y: bool = False,
    alpha_shuffle: float = 0.0,  # 0 = no shuffle, 1 = shuffle 100% of the data
    train_len: int = 60000,  # default MNIST train len
    test_len: int = 10000,  # default MNIST test len
    flatten: bool = False,  # flatten the images in the end
    size: int = 28,  # default MNIST size is 28x28
    num_workers: int = 0,
    **kwargs,
) -> tuple[BaseDataset, BaseDataset, DataLoader, DataLoader, list]:
    """Get the MNIST dataset.

    Parameters
    ----------
    dataset_path
        Path to the dataset
    batch_size, optional
        Train set batch size, by default 128
    batch_size_test, optional
        Test set batch size, by default 128
    noise_scale, optional
        Sigma value for the Gaussian noise added to the train set X, by default 
        0.0
    one_hot_encode_y, optional
        Boolean flag to one-hot encode y, by default False
    alpha_shuffle, optional
        The fraction of train samples that will be shuffled, by default 0.0
    train_len, optional
        Length of the train dataset, by default 60000
    test_len, optional
        Length of the test dataset, by default 10000
    flatten, optinal
        Flattens each samples to a vector, by default False
    size, optional
        Resizes 28x28 MNIST images to size x size, by default 28
    num_workers, optional
        Number of workers, by default 0

    Returns
    -------
        Returns the values in the following order: `train_dataset`, 
        `test_dataset`, `train_dataloader`, `test_dataloader`, `dataset_dims`
    """
    train_dataset = MNISTWrapper(
        dataset_path,
        True,
        noise_scale,
        one_hot_encode_y,
        train_len,
        alpha_shuffle,
        flatten,
        size,
        **kwargs,
    )
    test_dataset = MNISTWrapper(
        dataset_path,
        False,
        0.0,
        one_hot_encode_y,
        test_len,
        0.0,
        flatten,
        size,
        **kwargs,
    )

    g = torch.Generator()
    g.manual_seed(0)

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        worker_init_fn=seed_worker,
        generator=g,
    )
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size_test,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        worker_init_fn=seed_worker,
        generator=g,
    )

    if flatten:
        dims = [size * size, 10]
    else:
        dims = [[1, size, size], 10]

    return train_dataset, test_dataset, train_dataloader, test_dataloader, dims
