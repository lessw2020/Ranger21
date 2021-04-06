# Ranger21 - @lessw2020

# core components based on:

# MADGRAD: https://arxiv.org/abs/2101.11075

# warmup:  https://arxiv.org/abs/1910.04209v3

# stable weight decay: https://arxiv.org/abs/2011.11152v3

# Gradient Centralization: https://arxiv.org/abs/2004.01461v2


import torch
import torch.optim as TO

import math
import collections

import copy
from torch import linalg as LA


class Ranger21(TO.Optimizer):
    def __init__(
        self,
        params,
        lr,
        betas=(0.9, 0.999),  # temp for checking tuned warmups
        momentum=0.9,
        eps=1e-8,
        num_batches_per_epoch=None,
        num_epochs=None,
        num_warmup_iterations=1000,
        weight_decay=0,
        decay_type="stable",
        warmup_type="linear",
        use_GC=True,
    ):

        # todo - checks on incoming params
        defaults = dict(
            lr=lr, momentum=momentum, betas=betas, eps=eps, weight_decay=weight_decay
        )
        super().__init__(params, defaults)

        self.num_batches = num_batches_per_epoch
        self.num_epochs = num_epochs
        self.num_warmup_iters = num_warmup_iterations
        self.warmup_type = warmup_type
        self.use_GC = use_GC
        self.starting_lr = lr

        # decay
        self.decay = weight_decay
        self.decay_type = decay_type
        self.param_size = 0

        # logging
        self.variance_sum_tracking = []

    def __setstate__(self, state):
        super().__setstate__(state)

    def warmup_dampening(self, step):
        # not usable yet
        style = self.warmup_type
        step += 1
        warmup = self.num_warmup_iters

        if style is None:
            return 1.0

        if style == "linear":
            return min(1.0, (step / warmup))

        elif style == "exponential":
            return 1.0 - math.exp(-step / warmup)
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
        # if closure is not None and isinstance(closure, collections.Callable):
        #    with torch.grad():
        #        loss = closure()

        # if closure is not None:
        #    with torch.enable_grad():
        #        loss = closure()

        param_size = 0
        variance_ma_sum = 0.0

        for i, group in enumerate(self.param_groups):
            for j, p in enumerate(group["params"]):
                if p.grad is None:
                    continue

                if not self.param_size:
                    param_size += p.numel()

                # Perform optimization step
                grad = p.grad

                if grad.is_sparse:
                    raise RuntimeError("sparse matrix not supported atm")

                state = self.state[p]

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

                state["step"] += 1

                beta1, beta2 = group["betas"]
                grad_ma = state["grad_ma"]
                variance_ma = state["variance_ma"]

                bias_correction2 = 1 - beta2 ** state["step"]

                # update the exp averages
                grad_ma.mul_(beta1).add_(grad, alpha=1 - beta1)

                variance_ma.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                variance_ma_debiased = variance_ma / bias_correction2

                variance_ma_sum += variance_ma_debiased.sum()

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

            variance_normalized = math.sqrt(variance_ma_sum / self.param_size)

            # print(f"variance mean sqrt = {variance_normalized}")

        # phase 2 - apply weight decay and step
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]

                step = state["step"]

                # Perform stable weight decay
                decay = group["weight_decay"]
                eps = group["eps"]
                lr = group["lr"]

                if decay:
                    p.data.mul_(1 - decay * lr / variance_normalized)

                beta1, beta2 = group["betas"]
                grad_exp_avg = state["grad_ma"]
                variance_ma = state["variance_ma"]

                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step

                variance_biased_ma = variance_ma / bias_correction2

                denom = variance_biased_ma.sqrt().add(eps)

                step_size = lr / bias_correction1

                # update weights
                p.addcdiv_(grad_exp_avg, denom, value=-step_size)

        return loss

    """    param_size = 0
            variance_avg_sum = 0.

            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    param_size += p.numel()

                    # first part of optimization
                    grad = p.grad

                    

                    state = self.state[p]

                    #init if needed
                    if len(state)==0:
                        print(f"initing state")
                        state['step']=0

                        state['mean_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state['variance_avg'] = torch.zeros_like(p,memory_format = torch.preserve_format)

                    # get state values
                    #beta1, beta2, mean_avg, variance_avg = self.get_state_values(group, state)
                    beta1,beta2 = group['betas']
                    mean_avg = state['mean_avg']
                    variance_avg = state['variance_avg']

                    #print(f"beta1= {beta1}")

                    state['step'] +=1

                    bias_correction2 = 1 - beta2**variance_avg

                    # bias  avgs
                    mean_avg.mul_(beta1).add_(grad, alpha=1-beta1)
                    variance_avg.mul_(beta2).addcmul_(grad,grad,value = 1-beta2)

                    variance_avg_hat = variance_avg / bias_correction2
                    #print(f"variance-avg-hat = {variance_avg_hat}")

                    variance_avg_sum += variance_avg_hat.sum()

                print(f"param size = {param_size}")
                if not self.paramsize:
                    self.paramsize = param_size
                else:
                    if self.paramsize != param_size:
                        raise ValueError("param size changed")
                print(f"variance_avg_sum = {variance_avg_sum}")
                variance_avg_normalized = math.sqrt(variance_avg_sum / param_size)
                print(f"variance sum normalize = {variance_avg_normalized}")
"""
