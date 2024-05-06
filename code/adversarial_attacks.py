from collections.abc import Callable

import torch


def adverse_batch_pgd(
    x_batch: torch.Tensor,
    y_batch: torch.Tensor,
    model: torch.nn.Module,
    ord: int = 2,
    epsilon: float = 0.01,
    init_noise: float = 1e-4,
    max_steps: int = 1000,
    gamma: float = 10.0,
    gamma_decay: float = 0.95,
    loss_f: Callable = torch.nn.CrossEntropyLoss(reduction="mean"),
    seed: int = 0,
    verbose: bool = False,
):
    """A PGD attack that is applied to a batch of points.

    Parameters
    ----------
    x_batch
        A batch of input points.
    y_batch
        A batch of labels.
    model
        Model object.
    ord, optional
        The norm of the attack. Only 2 and torch.inf norms are supported, by 
        default 2.
    epsilon, optional
        The epsilon parameter of the attack, by default 0.01.
    init_noise, optional
        STD of the gaussian noise added to inputs, by default 1e-4.
    max_steps, optional
        Maximum number of iteration steps, by default 1000.
    gamma, optional
        Learning rate of the algorithm, by default 10.0.
    gamma_decay, optional
        Decay parameter for the learning rate, by default 0.95.
    loss_f, optional
        The loss function, by default 
        torch.nn.CrossEntropyLoss(reduction="mean").
    seed, optional
        Random seed, by default 0.
    verbose, optional
        Verbose logging level, by default False.

    Returns
    -------
        A batch of adversarially perturbed points.
    """
    torch.manual_seed(seed)

    assert 0.0 < gamma_decay <= 1.0

    x_flat = torch.clone(x_batch).flatten(1)

    # start with some random noise
    noise = torch.normal(mean=0.0, std=torch.ones_like(x_flat) * init_noise)
    xtp1 = (x_flat + noise).detach()

    for step in range(max_steps):
        if verbose:
            print(f"Iter {step+1}/{max_steps}...")
        xt = torch.clone(xtp1.detach())

        xt.requires_grad_(True)

        # compute grad of the loss
        loss = loss_f(model(xt.reshape(x_batch.shape)), y_batch)
        loss.backward()

        # norm grad
        grad_x_batch = xt.grad.data.flatten(1)
        if ord == 2:
            grad_x_batch = torch.nn.functional.normalize(grad_x_batch,
                                                         p=2,
                                                         dim=1)
        if ord == torch.inf:
            grad_x_batch = torch.sign(grad_x_batch)

        # step
        xtp1 = xt + gamma * grad_x_batch
        gamma *= gamma_decay

        # project to an L_ord ball
        if ord == 2:
            delta = xtp1 - x_flat
            delta_norm = torch.linalg.norm(delta, ord=2, dim=1)

            mask = delta_norm >= epsilon
            scaling = delta_norm
            scaling[mask] = epsilon

            # repeat the scaling batched constant array so that pytorch can
            # multiply it with batched vector
            scaling = scaling.unsqueeze(1).repeat(1, delta.shape[1])

            delta = torch.nn.functional.normalize(delta, p=2, dim=1) * scaling

            xtp1 = x_flat + delta
        if ord == torch.inf:
            xtp1 = torch.clamp(xtp1, x_flat - epsilon, x_flat + epsilon)

        # project to the L inf domain box
        if ord == 2:
            mask_to_proj = (torch.min(xtp1, dim=1)[0] < 0.0) + \
                (torch.max(xtp1, dim=1)[0] > 1.0)
            if torch.sum(mask_to_proj) > 0:
                t = proj_to_box(x_flat[mask_to_proj],
                                xtp1[mask_to_proj], epsilon)
                xtp1[mask_to_proj] = t
        if ord == torch.inf:
            xtp1 = torch.clamp(xtp1, 0.0, 1.0)

    if verbose:
        iter_gap = torch.linalg.norm(xtp1 - xt, ord=2, dim=1)
        print(f"Last iteration = {step+1}, iteration gaps = {iter_gap}")

    return xtp1.reshape(x_batch.shape).detach()


def proj_to_box(
    x_batch: torch.Tensor, x_batch_proj_ball: torch.Tensor, epsilon: float
) -> torch.Tensor:
    """Projects the point batch to a 0-1 box according to the L2 norm.

    Parameters
    ----------
    x_batch
        Original batch of unporjected points.
    x_batch_proj_ball
        A batch of points that are projected to the L2 ball, ready to be 
        projected to the L2 box.
    epsilon
        The size of the L2 ball.

    Returns
    -------
        A batch of points that are projected to the L2 box.
    """
    x_batch_proj_ball_ = torch.clamp(x_batch_proj_ball, min=0.0, max=1.0)

    hdir_batch = (
        torch.nn.functional.normalize(
            x_batch_proj_ball - x_batch_proj_ball_, p=2, dim=1) * epsilon
    )
    h_batch_proj_ball = x_batch + hdir_batch
    h_batch_proj_ball_ = torch.clamp(h_batch_proj_ball, min=0.0, max=1.0)

    # add 1e-20 to combat division by zero
    frac_batch = torch.linalg.norm(
        x_batch - h_batch_proj_ball_, ord=2, dim=1) / (
        torch.linalg.norm(x_batch_proj_ball -
                          x_batch_proj_ball_, ord=2, dim=1) + 1e-20
    )
    delta_batch = frac_batch * epsilon / (1 + frac_batch)

    # repeat the scaling batched constant array so that pytorch can multiply it
    # with batched vector
    delta_batch = delta_batch.unsqueeze(1).repeat(1, x_batch.shape[1])

    return (
        x_batch
        + torch.nn.functional.normalize(x_batch_proj_ball -
                                        x_batch, p=2, dim=1) * delta_batch
    )
