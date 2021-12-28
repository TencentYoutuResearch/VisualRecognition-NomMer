# -*- encoding: utf-8 -*-
# ----------------------------------------------
# filename        :samplers.py
# description     :NomMer: Nominate Synergistic Context in Vision Transformer for Visual Recognition
# date            :2021/12/28 17:45:58
# author          :clark
# version number  :1.0
# ----------------------------------------------


import torch

class SubsetRandomSampler(torch.utils.data.Sampler):
    r"""Samples elements randomly from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.epoch = 0
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in torch.randperm(len(self.indices)))

    def __len__(self):
        return len(self.indices)

    def set_epoch(self, epoch):
        self.epoch = epoch
