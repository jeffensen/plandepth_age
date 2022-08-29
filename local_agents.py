#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
from torch import zeros, ones, randn
zeros = torch.zeros
ones = torch.ones
randn = torch.randn


class MFRL(object):
    def __init__(self):
        pass

    def set_parameters(self, trans_par=None, true_params=False):
        # response probability
        self.logits = []

    def update_beliefs(self, block, trial, states, conditions, responses=None):
        raise NotImplementedError

    def plan_actions(self, block, trial):

        raise NotImplementedError
        
    def sample_responses(self, block, trial):
        raise NotImplementedError
