"""Learning rate schedulers for PyTorch optimizers."""
from math import ceil, floor

import torch


def const(optim: torch.optim.Optimizer, batch_size: int, dataset_len: int):
    return torch.optim.lr_scheduler.ConstantLR(optim, factor=1)


def step25(optim: torch.optim.Optimizer, batch_size: int, dataset_len: int):
    updates_per_epoch = ceil(dataset_len / batch_size)
    return step_scheduler(optim, perc=0.25, updates=10_000 * updates_per_epoch,
                          gamma=0.75)


def step10(optim: torch.optim.Optimizer, batch_size: int, dataset_len: int):
    updates_per_epoch = ceil(dataset_len / batch_size)
    return step_scheduler(optim, perc=0.10, updates=3_000 * updates_per_epoch,
                          gamma=0.75)


def warmup20000Step25(optim: torch.optim.Optimizer, batch_size: int,
                      dataset_len: int):
    updates_per_epoch = ceil(dataset_len / batch_size)
    return linear_and_step_scheduler(
        optim,
        linear_updates=20000,  # 20000 updates warmup
        step_perc=0.25,
        step_updates=10_000 * updates_per_epoch,  # 10000 epochs
        step_gamma=0.75,
    )


def cont100(
    optim: torch.optim.Optimizer,
    batch_size: int,
    dataset_len: int,
    epochs: int = 100,
    gamma: float = 0.95,
):
    updates_per_epoch = ceil(dataset_len / batch_size)
    return torch.optim.lr_scheduler.StepLR(
        optim, gamma=gamma, step_size=epochs * updates_per_epoch, last_epoch=-1
    )


def cont100_limit_lr(
    optim: torch.optim.Optimizer,
    batch_size: int,
    dataset_len: int,
    epochs: int = 100,
    gamma: float = 0.95,
    min_lr_factor: float = 1e-3,
):
    updates_per_epoch = ceil(dataset_len / batch_size)

    def lr_lambda(step): return max(
        gamma ** floor(step / (updates_per_epoch * epochs)), min_lr_factor)
    return torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_lambda,
                                             last_epoch=-1)


def warmup20000Cont2500(optim: torch.optim.Optimizer, batch_size: int,
                        dataset_len: int):
    updates_per_epoch = ceil(dataset_len / batch_size)
    return linear_and_step_scheduler_cont(
        optim,
        linear_updates=20000,  # 20000 updates warmup
        step_updates=2500 * updates_per_epoch,
        step_gamma=0.95,
    )


# Helpers


def step_scheduler(
    optim: torch.optim.Optimizer, perc: float = 0.25, updates: int = 10_000 * 8,
    gamma: float = 0.75
):
    n = int(1 / perc)
    milestones = [int(perc * i * updates) for i in range(1, n)]
    return torch.optim.lr_scheduler.MultiStepLR(
        optim,
        milestones=milestones,
        gamma=gamma,
    )


def linear_and_step_scheduler(
    optim: torch.optim.Optimizer,
    linear_updates: int = 10,
    step_perc: float = 0.25,
    step_updates: int = 10_000 * 8,
    step_gamma: float = 0.75,
):
    return torch.optim.lr_scheduler.SequentialLR(
        optim,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(
                optim,
                start_factor=1 / linear_updates,
                # increase by 1/n for n updates (from 0 to n-1)
                total_iters=linear_updates - 1,
            ),
            step_scheduler(optim, step_perc, step_updates, step_gamma),
        ],
        # after n+1 updates, switch to the second scheduler
        milestones=[linear_updates + 1],
    )


def linear_and_step_scheduler_cont(
    optim: torch.optim.Optimizer,
    linear_updates: int = 10,
    step_updates: int = 10_000 * 8,
    step_gamma: float = 0.75,
):
    return torch.optim.lr_scheduler.SequentialLR(
        optim,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(
                optim,
                start_factor=1 / linear_updates,
                # increase by 1/n for n updates (from 0 to n-1)
                total_iters=linear_updates - 1,
            ),
            torch.optim.lr_scheduler.StepLR(
                optim, gamma=step_gamma, step_size=step_updates, last_epoch=-1
            ),
        ],
        # after n+1 updates, switch to the second scheduler
        milestones=[linear_updates + 1],
    )
