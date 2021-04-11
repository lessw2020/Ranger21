# Ranger21 - @lessw2020

# core components based on:

# MADGRAD: https://arxiv.org/abs/2101.11075

# warmup:  https://arxiv.org/abs/1910.04209v3

# stable weight decay: https://arxiv.org/abs/2011.11152v3

# Gradient Centralization: https://arxiv.org/abs/2004.01461v2


import torch
import torch.optim as TO
import torch.nn.functional as F

import math
import collections

import copy
from torch import linalg as LA


def centralize_gradient(x, gc_conv_only=False):
    """credit - https://github.com/Yonghongwei/Gradient-Centralization """

    size = len(list(x.size()))
    # print(f"size = {size}")

    if gc_conv_only:
        if size > 3:
            x.add_(-x.mean(dim=tuple(range(1, size)), keepdim=True))
    else:
        if size > 1:
            x.add_(-x.mean(dim=tuple(range(1, size)), keepdim=True))
    return x


class Ranger21(TO.Optimizer):
    def __init__(
        self,
        params,
        lr,
        use_madgrad=True,
        using_gc=True,
        gc_conv_only=False,
        betas=(0.9, 0.999),  # temp for checking tuned warmups
        momentum=0.9,
        eps=1e-8,
        num_batches_per_epoch=None,
        num_epochs=None,
        use_warmup=True,
        num_warmup_iterations=None,
        weight_decay=1e-4,
        decay_type="stable",
        warmup_type="linear",
    ):

        # todo - checks on incoming params
        defaults = dict(
            lr=lr, momentum=momentum, betas=betas, eps=eps, weight_decay=weight_decay
        )
        super().__init__(params, defaults)

        # engine
        self.use_madgrad = use_madgrad
        self.num_batches = num_batches_per_epoch
        self.num_epochs = num_epochs

        self.warmup_type = warmup_type
        self.use_gc = using_gc
        self.gc_conv_only = gc_conv_only
        self.starting_lr = lr

        # decay
        self.decay = weight_decay
        self.decay_type = decay_type
        self.param_size = 0

        # warmup - we'll use default recommended in Ma/Yarats unless user specifies num iterations
        self.use_warmup = use_warmup

        if num_warmup_iterations is None:
            self.num_warmup_iters = math.ceil(
                (2 / (1 - betas[1]))
            )  # default untuned linear warmup
        else:
            self.num_warmup_iters = num_warmup_iterations

        # logging
        self.variance_sum_tracking = []

        # display
        engine = "Adam" if not self.use_madgrad else "MadGrad"

        # print out initial settings to make usage easier
        print(f"Ranger21 optimizer ready with following settings:\n")
        print(f"Core optimizer = {engine}")
        print(f"Learning rate of {self.starting_lr}")

        if self.use_warmup:
            print(f"{self.warmup_type} warmup, over {self.num_warmup_iters} iterations")
        if self.decay:
            print(f"Stable weight decay of {self.decay}")

        if self.use_gc:
            print(f"Gradient Centralization = On")
        else:
            print("Gradient Centralization = Off")

    def __setstate__(self, state):
        super().__setstate__(state)

    def warmup_dampening(self, lr, step):
        # not usable yet
        style = self.warmup_type
        warmup = self.num_warmup_iters

        if style is None:
            return 1.0

        if style == "linear":
            return lr * min(1.0, (step / warmup))

        elif style == "exponential":
            return lr * (1.0 - math.exp(-step / warmup))
        else:
            raise ValueError(f"warmup type {style} not implemented.")

    def get_variance(self):
        return self.variance_sum_tracking

    def get_state_values(self, group, state):
        beta1, beta2 = group["betas"]
        mean_avg = state["mean_avg"]
        variance_avg = state["variance_avg"]

        return beta1, beta2, mean_avg, variance_avg

    # @staticmethod
    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None and isinstance(closure, collections.Callable):
            with torch.grad():
                loss = closure()

        param_size = 0
        variance_ma_sum = 0.0

        # phase 1 - accumulate all of the variance_ma_sum to use in stable weight decay

        for i, group in enumerate(self.param_groups):
            for j, p in enumerate(group["params"]):
                if p.grad is None:
                    continue

                if not self.param_size:
                    param_size += p.numel()

                grad = p.grad

                if grad.is_sparse:
                    raise RuntimeError("sparse matrix not supported atm")

                state = self.state[p]
                momentum = group["momentum"]

                # State initialization
                if len(state) == 0:
                    # print("init state")
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["grad_ma"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    # Exponential moving average of squared gradient values
                    state["variance_ma"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                # centralize gradients
                if self.use_gc:
                    grad = centralize_gradient(
                        grad,
                        gc_conv_only=self.gc_conv_only,
                    )
                # else:
                #    grad = uncentralized_grad

                state["step"] += 1

                beta1, beta2 = group["betas"]
                grad_ma = state["grad_ma"]

                variance_ma = state["variance_ma"]
                step = state["step"]
                lr = group["lr"]

                # if self.use_warmup:
                #    lr = self.warmup_dampening(lr, step)

                bias_correction2 = 1 - beta2 ** state["step"]

                # update the exp averages
                # if not self.use_madgrad:
                grad_ma.mul_(beta1).add_(grad, alpha=1 - beta1)

                variance_ma.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                variance_ma_debiased = variance_ma / bias_correction2

                variance_ma_sum += variance_ma_debiased.sum()
                # else: #madgrad

                # should we dupe variance_ma since stable is assuming adam style variance?

                # stable wd
                # variance_ma_sum += grad_sum_sq.sum()

            # print(f"variance hat sum = {exp_avg_sq_hat_sum}")
            # Calculate the sqrt of the mean of all elements in exp_avg_sq_hat

            # we will run this first epoch only and then memoize
            if not self.param_size:
                self.param_size = param_size
                print(f"params size saved")
                print(f"total param groups = {i+1}")
                print(f"total params in groups = {j+1}")

            if not self.param_size:
                raise ValueError("failed to set param size")

            # debugging
            self.variance_sum_tracking.append(variance_ma_sum.item())

            # stable weight decay
            # if not self.use_madgrad:
            variance_normalized = math.sqrt(variance_ma_sum / self.param_size)
            # else:
            #    variance_normalized = math.pow((variance_ma / self.param_size), .3333)

            # print(f"variance mean sqrt = {variance_normalized}")

        # phase 2 - apply weight decay and step
        for group in self.param_groups:

            step = state["step"]

            # Perform stable weight decay
            decay = group["weight_decay"]
            eps = group["eps"]
            lr = group["lr"]
            momentum = group["momentum"]

            beta1, beta2 = group["betas"]
            grad_exp_avg = state["grad_ma"]
            variance_ma = state["variance_ma"]

            if self.use_warmup:
                lr = self.warmup_dampening(lr, step)

            # madgrad outer
            ck = 1 - momentum
            lamb = lr * math.pow(step, 0.5)

            if decay:
                if not self.use_madgrad:
                    p.data.mul_(1 - decay * lr / variance_normalized)
                else:
                    p.data.mul_(1 - decay * lamb / variance_normalized)

            # innner loop, params
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                grad = p.grad


                if self.use_madgrad:
                    if "grad_sum_sq" not in state:
                        state["grad_sum_sq"] = torch.zeros_like(p.data).detach()
                        state["s"] = torch.zeros_like(p.data).detach()
                        if momentum != 0:
                            state["x0"] = torch.clone(p.data).detach()

                    if momentum != 0.0 and grad.is_sparse:
                        raise RuntimeError(
                            "momentum != 0 is not compatible with sparse gradients"
                        )

                    # centralize gradients
                    if self.use_gc:
                        grad = centralize_gradient(
                            grad,
                            gc_conv_only=self.gc_conv_only,
                        )

                    grad_sum_sq = state["grad_sum_sq"]
                    s = state["s"]
                    if momentum == 0:
                        # Compute x_0 from other known quantities
                        rms = grad_sum_sq.pow(1 / 3).add_(eps)
                        x0 = p.data.addcdiv(s, rms, value=1)
                    else:
                        x0 = state["x0"]

                    # Accumulate second moments

                    # print(f" grad = {grad}")
                    # print(f"lamb = {lamb}")
                    # print(f"gsumsq = {grad_sum_sq}")

                    grad_sum_sq.addcmul_(grad, grad, value=lamb)
                    rms = grad_sum_sq.pow(1 / 3).add_(eps)

                    # Update s
                    s.data.add_(grad, alpha=lamb)

                    # Step
                    if momentum == 0:
                        p.data.copy_(x0.addcdiv(s, rms, value=-1))
                    else:
                        z = x0.addcdiv(s, rms, value=-1)

                        # p is a moving average of z
                        p.data.mul_(1 - ck).add_(z, alpha=ck)

                else:  # adam core

                    beta1, beta2 = group["betas"]
                    grad_exp_avg = state["grad_ma"]
                    variance_ma = state["variance_ma"]
                    
                    # grad centralization, if used, was already used in the phase 1 pass as part of grad_exp_avg and variance_ma computations ...so no need to do it again here
                    bias_correction1 = 1 - beta1 ** step
                    bias_correction2 = 1 - beta2 ** step

                    variance_biased_ma = variance_ma / bias_correction2

                    denom = variance_biased_ma.sqrt().add(eps)

                    step_size = lr / bias_correction1

                    # update weights
                    # p.data.add_(weight_mod, alpha=-step_size)
                    p.addcdiv_(grad_exp_avg, denom, value=-step_size)

        return loss
