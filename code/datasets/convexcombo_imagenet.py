"""ImageNet dataset wrapper with convex combinations of two images."""
import itertools
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from datasets.imagenet import ImageNetWrapper
from torch.utils.data import DataLoader
from utils.functions import seed_worker


class ConvexComboImageNetWrapper(ImageNetWrapper):
    def __init__(
        self,
        dataset_path: Path,
        is_train: bool,
        one_hot_y: bool,
        dataset_len: int,
        dataset_start_index: int = 0,
        alpha_shuffle: float = 0.0,
        convex_lambda: float = 0.5,
        convex_indices_filepath: Path = None,
        size: int = 224,
        **kwargs,
    ):
        # noise scale is not supported
        super(ConvexComboImageNetWrapper, self).__init__(
            dataset_path=dataset_path,
            is_train=is_train,
            noise_scale=0.0,
            one_hot_y=one_hot_y,
            dataset_len=dataset_len,
            dataset_start_index=dataset_start_index,
            alpha_shuffle=alpha_shuffle,
            size=size,
            **kwargs,
        )
        # in the case of imagenet, we do not preprocess the data and save it on
        # disk, but keep on-the-fly transformations
        np.random.seed(27)

        self.dataset_len = dataset_len
        self.dataset_start_index = dataset_start_index

        if convex_indices_filepath is None:
            # use normal indices
            self.x_indices2 = np.random.choice(len(self.data),
                                               size=len(self.data),
                                               replace=False)[
                dataset_start_index: dataset_start_index + dataset_len
            ]
            self.convex_indices = np.vstack(
                [self.x_indices, self.x_indices2]).T
        else:
            indices_to_use = np.load(convex_indices_filepath)
            assert dataset_len <= len(indices_to_use) ** 2
            indices_to_use_cartesian = np.array(
                list(itertools.product(indices_to_use, indices_to_use))
            )

            shuffle = np.arange(len(indices_to_use_cartesian))
            np.random.shuffle(shuffle)

            self.convex_indices = indices_to_use_cartesian[shuffle]
            self.convex_indices = self.convex_indices[
                dataset_start_index: dataset_start_index + dataset_len
            ]

        assert 0.0 < convex_lambda < 1.0
        self.convex_lambda = convex_lambda

    def __getitem__(self, idx):
        # since transforms (including the addition of noise)
        # are applied when this function is called,
        # we add the manual seed here.
        # The seed depends on the dataset index, so that different noise is
        # applied to different images.
        torch.manual_seed(42 + idx)
        x_idx, x_idx2 = self.convex_indices[idx]
        y_idx = x_idx

        x = (
            self.convex_lambda * self.data[x_idx][0]
            + (1 - self.convex_lambda) * self.data[x_idx2][0]
        )
        # keep the label of the first point
        y = torch.tensor(self.data[y_idx][1]).type(self.y_type)

        return x, y


def load_datasets(
    dataset_path: Path,
    batch_size: int = 64,
    batch_size_test: int = 64,
    convex_lambda: float = 0.5,
    convex_indices_filepath: Path = None,
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
    convex_lambda, optional
        The fraction that is used to compute the convex combination, by default 
        0.5. Has to be between 0.0 and 1.0.
    convex_indices_filepath, optional
        The path to the indices numpy array to use to compute convex 
        combinations.
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
    train_dataset = ConvexComboImageNetWrapper(
        dataset_path,
        True,
        one_hot_encode_y,
        train_len,
        train_start,
        alpha_shuffle,
        convex_lambda,
        convex_indices_filepath,
        size,
        **kwargs,
    )
    test_dataset = ConvexComboImageNetWrapper(
        dataset_path,
        False,
        one_hot_encode_y,
        test_len,
        test_start,
        0.0,
        convex_lambda,
        convex_indices_filepath,
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
