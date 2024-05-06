"""ImageNet dataset wrapper."""
from collections.abc import Sequence
from math import floor
from pathlib import Path

import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
from utils.functions import seed_worker


class ImageNetWrapper(Sequence):
    def __init__(
        self,
        dataset_path: Path,
        is_train: bool,
        noise_scale: float,
        one_hot_y: bool,
        dataset_len: int,
        dataset_start_index: int = 0,
        alpha_shuffle: float = 0.0,
        size: int = 224,
        **kwargs,
    ):
        # in the case of imagenet, we do not preprocess the data and save it on
        # disk, but keep on-the-fly transformations
        np.random.seed(42)

        self.dataset_len = dataset_len
        self.dataset_start_index = dataset_start_index

        # basic transform
        transforms = [
            T.Resize(256),
            T.CenterCrop(size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]

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

        # ohe
        target_transform = None
        if one_hot_y:
            def target_transform(x): return self.onehot(x, n_labels=1000)

        if is_train:
            self.data = torchvision.datasets.ImageFolder(
                dataset_path / "train",
                transform=T.Compose(transforms),
                target_transform=target_transform,
            )
        else:
            self.data = torchvision.datasets.ImageFolder(
                dataset_path / "val",
                transform=T.Compose(transforms),
                target_transform=target_transform,
            )

        # sample subset
        subset_indices = np.random.choice(len(self.data),
                                          size=len(self.data),
                                          replace=False)[
            dataset_start_index: dataset_start_index + dataset_len
        ]
        self.x_indices = subset_indices
        self.y_indices = subset_indices

        self.y_type = torch.FloatTensor if one_hot_y else torch.LongTensor

        # shuffle
        if alpha_shuffle > 0.0:
            self.y_indices = self.incremental_shuffle(
                self.y_indices, alpha_shuffle)

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        # since transforms (including the addition of noise) are applied when
        # this function is called, we add the manual seed here. The seed depends
        # on the dataset index, so that different noise is applied to different
        # images.
        torch.manual_seed(42 + idx)
        x_idx = self.x_indices[idx]
        y_idx = self.y_indices[idx]
        return (self.data[x_idx][0],
                torch.tensor(self.data[y_idx][1]).type(self.y_type))

    def incremental_shuffle(self, y: np.array, alpha: float) -> np.array:
        """Shuffles alpha*100% of the y array.
        Shuffling is incremental, meaning that a dataset shuffled by alpha=0.5 
        contains the same shuffled mappings as in alpha=0.25 (plus extra ones).

        Parameters
        ----------
        y
            Numpy array of labels or label indices.
        alpha
            Float in range from 0 to 1, showing the fraction fo the data 
            shuffled.

        Returns
        -------
            Shuffled array
        """
        np.random.seed(42)
        num_swaps = floor(len(y) * alpha) // 2

        # final indices for the shuffled y
        final_indices = np.arange(len(y))

        # list of indices that tell which swaps we should perform
        swap_indices = np.arange(len(y))
        np.random.shuffle(swap_indices)

        # perform swaps
        # get left and right side indices
        left = swap_indices[::2][:num_swaps]
        right = swap_indices[1::2][:num_swaps]
        # set left side indices to the right ones and vice versa
        final_indices[left] = right
        final_indices[right] = left

        return y[final_indices]

    def onehot(self, y: int, n_labels: int = 10) -> torch.FloatTensor:
        """Performs one-hot encoding of a single y value. Used for torch dataset 
        transforms.

        Parameters
        ----------
        y
            integer value that will be one-hot encoded.
        n_labels
            Number of possible label values.

        Returns
        -------
            One-hot encoded y vector.
        """
        y_oh = torch.zeros((n_labels), dtype=torch.float32)
        y_oh[y] = 1
        return y_oh


def load_datasets(
    dataset_path: Path,
    batch_size: int = 64,
    batch_size_test: int = 64,
    noise_scale: float = 0.0,
    one_hot_encode_y: bool = True,
    alpha_shuffle: float = 0.0,  # 0.0 = no shuffle, 1.0 = shuffle 100%
    train_start: int = 0,
    train_len: int = 1_281_167,
    test_start: int = 0,
    test_len: int = 50_000,
    size: int = 224,  # default ImageNet size for ResNets is 224x224
    num_workers: int = 0,
    **kwargs,
) -> tuple[Sequence, Sequence, DataLoader, DataLoader, list]:
    """Get the ImageNet Dataset.
    For this to work, you need to download ImageNet and unpack the archive in 
    the `dataset_path` folder. It should contain the `imagenet` folder with 
    `train` and `val` subfolders.

    Parameters
    ----------
    dataset_path
        Path to the dataset
    batch_size, optional
        Train set batch size, by default 64
    batch_size_test, optional
        Test set batch size, by default 64
    noise_scale, optional
        Sigma value for the Gaussian noise added to the train set X, by default 
        0.0
    one_hot_encode_y, optional
        Boolean flag to one-hot encode y, by default True
    alpha_shuffle, optional
        The fraction of train samples that will be shuffled, by default 0.0
    train_start, optional
        Index of the first item in the train dataset, by default 0. Used for
        partitioning. Leave by default for regular use.
    train_len, optional
        Length of the train dataset, by default 1_281_167
    test_start, optional
        Index of the first item in the test dataset, by default 0. Used for 
        partitioning. Leave by default for regular use.
    test_len, optional
        Length of the test dataset, by default 50_000
    size, optional
        Resizes ImageNet images to 256 x 256, then crops them to size x size. 
        By default 224
    num_workers, optional
        Number of workers, by default 0

    Returns
    -------
        Returns the values in the following order: `train_dataset`, 
        `test_dataset`, `train_dataloader`, `test_dataloader`, `dataset_dims`
    """
    train_dataset = ImageNetWrapper(
        dataset_path,
        True,
        noise_scale,
        one_hot_encode_y,
        train_len,
        train_start,
        alpha_shuffle,
        size,
        **kwargs,
    )
    test_dataset = ImageNetWrapper(
        dataset_path, False, 0.0, one_hot_encode_y, test_len, test_start,
        0.0, size, **kwargs
    )

    g = torch.Generator()
    g.manual_seed(0)

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
    )
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size_test,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
    )

    return (train_dataset,
            test_dataset,
            train_dataloader,
            test_dataloader,
            [[3, size, size], 1000])
