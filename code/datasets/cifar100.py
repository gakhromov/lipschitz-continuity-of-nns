"""CIFAR-100 dataset wrapper."""
import logging
from pathlib import Path

import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from datasets.base_dataset import BaseDataset
from torch.utils.data import DataLoader
from utils.functions import seed_worker

CIFAR100_FINE_TO_COARSE = [
    4,  # apple
    1,  # aquarium_fish
    14,  # baby
    8,  # bear
    0,  # beaver
    6,  # bed
    7,  # bee
    7,  # beetle
    18,  # bicycle
    3,  # bottle
    3,  # bowl
    14,  # boy
    9,  # bridge
    18,  # bus
    7,  # butterfly
    11,  # camel
    3,  # can
    9,  # castle
    7,  # caterpillar
    11,  # cattle
    6,  # chair
    11,  # chimpanzee
    5,  # clock
    10,  # cloud
    7,  # cockroach
    6,  # couch
    13,  # crab
    15,  # crocodile
    3,  # cup
    15,  # dinosaur
    0,  # dolphin
    11,  # elephant
    1,  # flatfish
    10,  # forest
    12,  # fox
    14,  # girl
    16,  # hamster
    9,  # house
    11,  # kangaroo
    5,  # keyboard
    5,  # lamp
    19,  # lawn_mower
    8,  # leopard
    8,  # lion
    15,  # lizard
    13,  # lobster
    14,  # man
    17,  # maple_tree
    18,  # motorcycle
    10,  # mountain
    16,  # mouse
    4,  # mushroom
    17,  # oak_tree
    4,  # orange
    2,  # orchid
    0,  # otter
    17,  # palm_tree
    4,  # pear
    18,  # pickup_truck
    17,  # pine_tree
    10,  # plain
    3,  # plate
    2,  # poppy
    12,  # porcupine
    12,  # possum
    16,  # rabbit
    12,  # raccoon
    1,  # ray
    9,  # road
    19,  # rocket
    2,  # rose
    10,  # sea
    0,  # seal
    1,  # shark
    16,  # shrew
    12,  # skunk
    9,  # skyscraper
    13,  # snail
    15,  # snake
    13,  # spider
    16,  # squirrel
    19,  # streetcar
    2,  # sunflower
    4,  # sweet_pepper
    6,  # table
    19,  # tank
    5,  # telephone
    5,  # television
    8,  # tiger
    19,  # tractor
    18,  # train
    1,  # trout
    2,  # tulip
    15,  # turtle
    6,  # wardrobe
    0,  # whale
    17,  # willow_tree
    8,  # wolf
    14,  # woman
    13,  # worm
]


class CIFAR100Wrapper(BaseDataset):
    def __init__(
        self,
        dataset_path: Path,
        is_train: bool,
        noise_scale: float,
        one_hot_y: bool,
        dataset_len: int,
        alpha_shuffle: float = 0.0,
        greyscale: bool = False,
        flatten: bool = False,
        size: int = 32,
        num_labels: int = 100,
        **kwargs,
    ):
        super(CIFAR100Wrapper, self).__init__(
            dataset_path=dataset_path,
            is_train=is_train,
            noise_scale=noise_scale,
            one_hot_y=one_hot_y,
            dataset_len=dataset_len,
            alpha_shuffle=alpha_shuffle,
            greyscale=greyscale,
            flatten=flatten,
            size=size,
            num_labels=num_labels,
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
        greyscale: bool = False,
        flatten: bool = False,
        size: int = 32,
        num_labels: int = 100,
    ):
        np.random.seed(42)
        torch.manual_seed(42)

        full_ds_path = (
            dataset_path
            / "datasets"
            / (f"cifar100_{is_train}_{greyscale}_{flatten}_{size}_{num_labels}_"
               f"{one_hot_y}_{dataset_len}_{alpha_shuffle}.pt")
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

        if greyscale:
            transforms.append(T.Grayscale())
        transforms.append(T.Resize(size))
        if flatten:
            transforms.append(T.Lambda(torch.flatten))

        # change labels if needed
        def preprocess_labels(y): return y  # identity
        if num_labels == 20:
            def preprocess_labels(y): return CIFAR100_FINE_TO_COARSE[y]

        if one_hot_y:
            def preprocess_labels(y): return self.onehot(
                preprocess_labels(y), num_labels)

        # download the dataset
        dataset = torchvision.datasets.CIFAR100(
            str((dataset_path / "datasets").resolve()),
            train=is_train,
            transform=T.Compose(transforms),
            target_transform=preprocess_labels,
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
    train_len: int = 50000,  # default CIFAR100 train len
    test_len: int = 10000,  # default CIFAR100 test len
    greyscale: bool = False,  # convert from RGB to greyscale
    flatten: bool = False,  # flatten the images in the end
    size: int = 32,  # default CIFAR100 size is 32x32
    # number of classes: 100 (fine labels) or 20 (coarse labels)
    num_labels: int = 100,
    num_workers: int = 0,
    **kwargs,
) -> tuple[BaseDataset, BaseDataset, DataLoader, DataLoader, list]:
    """Get the CIFAR100 dataset.

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
        Length of the train dataset, by default 50000
    test_len, optional
        Length of the test dataset, by default 10000
    greyscale, optional
        Converts RGB samples to greyscale, reducing one dimension. By default 
        False
    flatten, optinal
        Flattens each samples to a vector, by default False
    size, optional
        Resizes 32x32 CIFAR images to size x size, by default 32
    num_labels, optional
        Number of classes: 100 (fine labels) or 20 (coarse labels), by default 
        100
    num_workers, optional
        Number of workers, by default 0

    Returns
    -------
        Returns the values in the following order: `train_dataset`, 
        `test_dataset`, `train_dataloader`, `test_dataloader`, `dataset_dims`
    """
    train_dataset = CIFAR100Wrapper(
        dataset_path,
        True,
        noise_scale,
        one_hot_encode_y,
        train_len,
        alpha_shuffle,
        greyscale,
        flatten,
        size,
        num_labels,
        **kwargs,
    )
    test_dataset = CIFAR100Wrapper(
        dataset_path,
        False,
        0.0,
        one_hot_encode_y,
        test_len,
        0.0,
        greyscale,
        flatten,
        size,
        num_labels,
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
        if greyscale:
            dims = [size * size, num_labels]
        else:
            dims = [3 * size * size, num_labels]
    else:
        if greyscale:
            dims = [[1, size, size], num_labels]
        else:
            dims = [[3, size, size], num_labels]

    return train_dataset, test_dataset, train_dataloader, test_dataloader, dims
