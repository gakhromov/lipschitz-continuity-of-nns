"""Base class for all dataset wrappers."""
from collections.abc import Iterable, Sequence
from math import floor
from pathlib import Path

import numpy as np
import torch


class BaseDataset(Sequence):
    def __init__(self, dataset_path: Path, **kwargs):
        ds_path = dataset_path / "datasets"
        if not ds_path.exists():
            ds_path.mkdir()

        self.baked_x, self.baked_y = self.bake_dataset(dataset_path, **kwargs)

    def bake_dataset(
        self, dataset_path: Path, **kwargs
    ) -> tuple[torch.FloatTensor, torch.FloatTensor | torch.LongTensor]:
        """This function prepares the dataset, loading the data and applying all
        the transforamtions.

        Parameters
        ----------
        dataset_path
            Path to the dataset

        Returns
        -------
            Return fully prepared `X`, `y` in the tensor format.
            `y` is of type `FloatTensor` if it is one-hot encoded, `LongTensor` 
            otherwise.

        Note
        ----
            This function has to be implemented in daughter classes.
        """
        # IMPLEMENT this function in the daughter class
        return torch.tensor(), torch.tensor()

    def __len__(self):
        return len(self.baked_y)

    def __getitem__(self, idx):
        return self.baked_x[idx], self.baked_y[idx]

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

    def shuffle(self, y: np.array, alpha: float) -> np.array:
        """Shuffles alpha*100% of the y array.

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
        # alpha - fraction of the data to be shuffled
        y_shuffled = y.copy()
        indices_to_shuffle = np.random.choice(
            range(len(y)), floor(len(y) * alpha), replace=False)
        shuffled_values = y_shuffled[indices_to_shuffle]
        np.random.shuffle(shuffled_values)
        y_shuffled[indices_to_shuffle] = shuffled_values
        return y_shuffled

    def onehot_vector(self, y: Iterable, n_labels: int = 10) -> np.array:
        """Performs one-hot encoding of y vector. Used for manual dataset 
        transformations.

        Parameters
        ----------
        y
            numpy array or torch tensor of integers that will be one-hot 
            encoded.
        n_labels
            Number of possible label values.

        Returns
        -------
            One-hot encoded y matrix.
        """
        y_oh = np.zeros((len(y), n_labels))
        indices = np.arange(len(y_oh))
        y_oh[indices, y] = 1
        return y_oh

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
