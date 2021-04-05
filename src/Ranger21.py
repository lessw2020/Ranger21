# Ranger21 - @lessw2020

# core components based on:

# MADGRAD: https://arxiv.org/abs/2101.11075

# warmup:  https://arxiv.org/abs/1910.04209v3

# stable weight decay: https://arxiv.org/abs/2011.11152v3

# Gradient Centralization: https://arxiv.org/abs/2004.01461v2


import torch
import torch.optim

import math
import collections

import copy
from torch import linalg as LA

class Ranger21(torch.optim.Optimizer):
    def __init__(self,
                params,
                lr,
                eps=1e-8,
                num_batches_per_epoch = None,
                num_epochs = None,
                num_warmup_iterations = 1000,
                weight_decay=0,
                decay_type = "stable",

                warmup_type = 'linear',
                use_GC=True):

                # todo - checks on incoming params
                defaults = dict(lr=lr, eps=eps, weight_decay = weight_decay)
                super().__init__(params, defaults)

                self.num_batches = num_batches_per_epoch
                self.num_epochs = num_epochs
                self.num_warmup_iters = num_warmup_iterations
                self.warmup_type=warmup_type
                self.use_GC = use_GC
                self.starting_lr = lr

                #decay
                self.decay = weight_decay
                self.decay_type = decay_type

    def warmup_dampening(self, step):
        # not usable yet
        style = self.warmup_type
        step +=1
        warmup = self.num_warmup_iters

        if style is None:
            return 1.0

        if style=='linear':
            return min(1.0, (step/warmup) )

        elif style=='exponential':
            return 1.0 - math.exp(-step/warmup)
        else:
            raise ValueError(f"warmup type {style} not implemented.")



    @torch.no_grad
    def step(self,
            closure = None,
            passed_loss = None):

            # let's build in a loss pass through for HyperExplorer
            loss = None
            if closure is not None and isinstance(closure, collections.Callable):
                with torch.grad():
                    loss = closure()
            



