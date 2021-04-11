# Ranger21 - @lessw2020
#  This is experimental branch of auto lr...not recommended for use atm.

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


class Ranger21abel(TO.Optimizer):
    def __init__(
        self,
        params,
        lr,
        betas=(0.9, 0.999),  # temp for checking tuned warmups
        momentum=0.9,
        eps=1e-8,
        num_batches_per_epoch=None,
        num_epochs=None,
        use_abel=True,
        abel_decay_factor = .3,
        use_warmup=True,
        num_warmup_iterations=None,
        weight_decay=1e-4,
        decay_type="stable",
        warmup_type="linear",
        use_gradient_centralization=True,
        gc_conv_only=False,
    ):

        # todo - checks on incoming params
        defaults = dict(
            lr=lr, momentum=momentum, betas=betas, eps=eps, weight_decay=weight_decay
        )
        super().__init__(params, defaults)

        self.num_batches = num_batches_per_epoch
        self.num_epochs = num_epochs

        self.warmup_type = warmup_type
        self.use_gc = (use_gradient_centralization,)
        self.gc_conv_only = (gc_conv_only,)
        self.starting_lr = lr
        self.current_lr = lr

        # abel
        self.use_abel = use_abel
        self.weight_list=[]
        self.batch_count =0
        self.epoch = 0
        self.lr_decay_factor = abel_decay_factor
        self.abel_decay_end = math.ceil(self.num_epochs * .85)
        self.reached_minima = False
        self.pweight_accumulator = 0
        
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

        

        # print out initial settings to make usage easier
        print(f"Ranger21 optimizer ready with following settings:\n")
        print(f"Learning rate of {self.starting_lr}")
        if self.use_warmup:
            print(f"{self.warmup_type} warmup, over {self.num_warmup_iters} iterations")

        print(f"Stable weight decay of {self.decay}")
        if self.use_gc:
            print(f"Gradient Centralization  = On")
        else:
            print("Gradient Centralization = Off")
        print(f"Num Epochs = {self.num_epochs}")
        print(f"Num batches per epoch = {self.num_batches}")

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

    def abel_update(self, step_fn, weight_norm, current_lr):
        ''' update lr based on abel'''
        
        self.pweight_accumulator += weight_norm

        
        self.batch_count +=1
        #print(f"self.batch count = {self.batch_count}")
        if self.batch_count == self.num_batches:
            self.epoch +=1
            self.batch_count = 0
            print(f"epoch eval for epoch {self.epoch}")

            #store weights
            self.weight_list.append(self.pweight_accumulator)
            
            print(f"total norm for epoch {self.epoch} = {weight_norm}")
            #self.pweight_accumulator = 0
        
        if self.batch_count !=0:
            return None
        #self.epoch +=1
        new_lr = current_lr

        if len(self.weight_list) < 3:
            print(len(self.weight_list))
            return step_fn
        
        # compute weight norm delta
        if (self.weight_list[-1] - self.weight_list[-2]) * (self.weight_list[-2] - self.weight_list[-3]) < 0:
            if self.reached_minima:
                self.reached_minima = False
                new_lr *= self.lr_decay_factor
                #step_fn = self.update_train_step(self.learning_rate)
            else:
                self.reached_minima = True
            print(f"\n*****\nABEL mininum detected, new lr = {new_lr}\n***\n")

        if self.epoch == self.abel_decay_end:
            new_lr *= self.lr_decay_factor
            print(f"abel final decay done, new lr = {new_lr}")
        return new_lr
    # @staticmethod
    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None and isinstance(closure, collections.Callable):
            with torch.grad():
                loss = closure()

        param_size = 0
        variance_ma_sum = 0.0
        weight_norm = 0


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

                current_weight_norm = LA.norm(p.data)
                #print(f"running norm = {current_weight_norm}")
                weight_norm += current_weight_norm.item()

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
                #lr = group["lr"]
                lr = self.current_lr

                if self.use_warmup:
                    lr = self.warmup_dampening(lr, step)
                    # if step < 10:
                    #    print(f"warmup dampening at step {step} = {lr} vs {group['lr']}")

                if decay:
                    p.data.mul_(1 - decay * lr / variance_normalized)

                beta1, beta2 = group["betas"]
                grad_exp_avg = state["grad_ma"]
                variance_ma = state["variance_ma"]

                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step

                variance_biased_ma = variance_ma / bias_correction2

                denom = variance_biased_ma.sqrt().add(eps)

                weight_mod = grad_exp_avg / denom
                
                step_size = lr / bias_correction1

                # update weights
                #p.data.add_(weight_mod, alpha=-step_size)
                p.addcdiv_(grad_exp_avg, denom, value=-step_size)

            # abel step
            abel_result = self.abel_update(None, weight_norm, self.current_lr)
            if abel_result is not None:
                self.current_lr = abel_result

        return loss
