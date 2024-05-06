"""List of enums used in training scripts."""
from enum import Enum
from functools import partial

import torch
import utils.lr_schedulers as lrs
from datasets import (cifar10, cifar100, convexcombo_cifar10,
                      convexcombo_imagenet, imagenet, mnist, mnist1d)


class Datasets(Enum):
    # all funcs in enums have to be wrapped in partial to behave correctly
    MNIST1D = partial(mnist1d.load_datasets)
    smallMNIST1D = partial(
        lambda *args, **kwargs: mnist1d.load_datasets(train_len=1000,
                                                      *args, **kwargs)
    )
    tinyMNIST1D = partial(
        lambda *args, **kwargs: mnist1d.load_datasets(train_len=500,
                                                      *args, **kwargs)
    )
    miniMNIST1D = partial(
        lambda *args, **kwargs: mnist1d.load_datasets(train_len=100,
                                                      *args, **kwargs)
    )

    MNIST = partial(mnist.load_datasets)
    miniMNIST = partial(
        lambda *args, **kwargs: mnist.load_datasets(train_len=5000,
                                                    *args, **kwargs)
    )

    CIFAR10 = partial(cifar10.load_datasets)
    miniCIFAR10 = partial(
        lambda *args, **kwargs: cifar10.load_datasets(train_len=5000,
                                                      *args, **kwargs)
    )
    minusculeCIFAR10 = partial(
        lambda *args, **kwargs: cifar10.load_datasets(train_len=100,
                                                      *args, **kwargs)
    )
    flatCIFAR10 = partial(
        lambda *args, **kwargs: cifar10.load_datasets(flatten=True,
                                                      *args, **kwargs)
    )
    ConvexComboCIFAR10 = partial(convexcombo_cifar10.load_datasets)

    CIFAR100 = partial(cifar100.load_datasets)
    CIFAR100c20 = partial(
        lambda *args, **kwargs: cifar100.load_datasets(num_labels=20,
                                                       *args, **kwargs)
    )
    miniCIFAR100c20 = partial(
        lambda *args, **kwargs: cifar100.load_datasets(
            num_labels=20, train_len=5000, *args, **kwargs
        )
    )

    ImageNet = partial(imagenet.load_datasets)
    ConvexComboImageNet = partial(convexcombo_imagenet.load_datasets)
    cutImageNet = partial(
        lambda *args, **kwargs: imagenet.load_datasets(train_len=8,
                                                       *args, **kwargs)
    )


class Devices(Enum):
    GPU = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CPU = torch.device("cpu")


class Losses(Enum):
    CrossEntropy = partial(
        lambda *args, **kwargs: torch.nn.CrossEntropyLoss(reduction="sum",
                                                          *args, **kwargs)
    )
    MSE = partial(lambda *args, **kwargs: torch.nn.MSELoss(reduction="sum",
                                                           *args, **kwargs))


class Optims(Enum):
    SGD = partial(torch.optim.SGD)
    Adam = partial(torch.optim.Adam)


class LR_Schedulers(Enum):
    # evert LR Scheduler call returns the scheduler function
    Const = partial(lrs.const)
    Step25 = partial(lrs.step25)
    Step10 = partial(lrs.step10)
    Warmup20000Step25 = partial(lrs.warmup20000Step25)
    Cont50 = partial(
        lambda optim, batch_size, dataset_len: lrs.cont100(
            optim, batch_size, dataset_len, epochs=50, gamma=0.95
        )
    )
    Cont100 = partial(lrs.cont100)
    Cont100LimitLR = partial(lrs.cont100_limit_lr)
    Cont200 = partial(
        lambda optim, batch_size, dataset_len: lrs.cont100(
            optim, batch_size, dataset_len, epochs=200, gamma=0.95
        )
    )
    Cont1000 = partial(
        lambda optim, batch_size, dataset_len: lrs.cont100(
            optim, batch_size, dataset_len, epochs=1000, gamma=0.95
        )
    )
    Warmup20000Cont2500 = partial(lrs.warmup20000Cont2500)
