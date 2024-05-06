"""MNIST1D dataset wrapper."""
import pickle
from pathlib import Path

import numpy as np
import requests
import torch
from datasets.base_dataset import BaseDataset
from torch.utils.data import DataLoader
from utils.functions import seed_worker


class MNIST1DWrapper(BaseDataset):
    def __init__(
        self,
        dataset_path: Path,
        is_train: bool,
        noise_scale: float,
        one_hot_y: bool,
        dataset_len: int,
        alpha_shuffle: float = 0.0,
        **kwargs,
    ):
        # pass arguments to the bake_dataset call
        super(MNIST1DWrapper, self).__init__(
            dataset_path=dataset_path,
            is_train=is_train,
            noise_scale=noise_scale,
            one_hot_y=one_hot_y,
            dataset_len=dataset_len,
            alpha_shuffle=alpha_shuffle,
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
        **kwargs,
    ) -> tuple[torch.FloatTensor, torch.FloatTensor | torch.LongTensor]:
        np.random.seed(42)
        ds_path = dataset_path / "datasets"

        # download the dataset
        if not (ds_path / "mnist1d_data.pkl").exists():
            url = ("https://github.com/greydanus/mnist1d/raw/master/"
                   "mnist1d_data.pkl")
            r = requests.get(url, allow_redirects=True)
            with (ds_path / "mnist1d_data.pkl").open(mode="wb") as f:
                f.write(r.content)

        with (ds_path / "mnist1d_data.pkl").open(mode="rb") as f:
            data = pickle.load(f)

        # retrieve raw data
        if is_train:
            x, y = data["x"], data["y"]
        else:
            x, y = data["x_test"], data["y_test"]

        # add noise
        if noise_scale > 0.0:
            noise = noise_scale * np.random.normal(size=x.shape)
            x = (x + noise).astype(np.float32)

        # one hot
        if one_hot_y:
            y = self.onehot_vector(y, 10)

        # sample subset
        subset_indices = np.random.choice(
            len(x), size=dataset_len, replace=False)
        x = x[subset_indices]
        y = y[subset_indices]

        # shuffle
        if alpha_shuffle > 0.0:
            y = self.incremental_shuffle(y, alpha_shuffle)

        # convert to tensor
        if one_hot_y:
            # if one_hot = True => MSE loss => float tensor
            return torch.Tensor(x), torch.Tensor(y).type(torch.FloatTensor)
        # if one_hot = False => CE loss => long tensor
        return torch.Tensor(x), torch.Tensor(y).type(torch.LongTensor)


def load_datasets(
    dataset_path: Path,
    batch_size: int = 512,
    batch_size_test: int = 512,
    noise_scale: float = 0.0,
    one_hot_encode_y: bool = True,
    alpha_shuffle: float = 0.0,  # 0.0 = no shuffle, 1.0 = shuffle 100%
    train_len: int = 4000,
    test_len: int = 1000,
    num_workers: int = 0,
    **kwargs,
) -> tuple[BaseDataset, BaseDataset, DataLoader, DataLoader, list]:
    """Get the MNIST1D dataset.

    Parameters
    ----------
    dataset_path
        Path to the dataset
    batch_size, optional
        Train set batch size, by default 512
    batch_size_test, optional
        Test set batch size, by default 512
    noise_scale, optional
        Sigma value for the Gaussian noise added to the train set X, by default 
        0.0
    one_hot_encode_y, optional
        Boolean flag to one-hot encode y, by default True
    alpha_shuffle, optional
        The fraction of train samples that will be shuffled, by default 0.0
    train_len, optional
        Length of the train dataset, by default 4000
    test_len, optional
        Length of the test dataset, by default 1000
    num_workers, optional
        Number of workers, by default 0

    Returns
    -------
        Returns the values in the following order: `train_dataset`, 
        `test_dataset`, `train_dataloader`, `test_dataloader`, `dataset_dims`
    """

    train_dataset = MNIST1DWrapper(
        dataset_path, True, noise_scale, one_hot_encode_y, train_len,
        alpha_shuffle, **kwargs
    )
    test_dataset = MNIST1DWrapper(
        dataset_path, False, 0.0, one_hot_encode_y, test_len, 0.0, **kwargs
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

    return (train_dataset,
            test_dataset,
            train_dataloader,
            test_dataloader,
            [40, 10])
