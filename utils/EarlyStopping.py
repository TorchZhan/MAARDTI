#!/usr/bin/env python
# coding=utf-8
'''
Author: Zhan
Date: 2023-08-02 17:54:35
'''
# import numpy as np
# import torch
# from datetime import datetime

# from rich.console import Console

# class EarlyStopping:
#     """Early stops the training if validation loss doesn't improve after a given patience."""
#     def __init__(self, savepath = None, patience=7, verbose=False, delta=0, num_n_fold = 0):
#         """
#         Args:
#             patience (int): How long to wait after last time validation loss improved.
#                             Default: 7
#             verbose (bool): If True, prints a message for each validation loss improvement.
#                             Default: False
#             delta (float): Minimum change in the monitored quantity to qualify as an improvement.
#                             Default: 0
#         """
#         self.console = Console()

#         self.patience = patience
#         self.verbose = verbose
#         self.counter = 0
#         self.best_score = None
#         self.early_stop = False
#         self.val_loss_min = np.Inf
#         self.delta = delta
#         self.num_n_fold = num_n_fold
#         self.savepath = savepath
#         self.current_time = datetime.now().strftime('%b%d_%H-%M-%S')

#     def __call__(self, val_loss, model, num_epoch):

#         score = -val_loss

#         if self.best_score is None:
#             self.best_score = score
#             self.save_checkpoint(val_loss, model,num_epoch)
#         elif score < self.best_score + self.delta:
#             self.counter += 1
            
#             # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
#             self.console.log(f'[red]    => EarlyStopping counter: {self.counter} out of {self.patience}')

#             if self.counter >= self.patience:
#                 self.early_stop = True
#         else:
#             self.best_score = score
#             self.save_checkpoint(val_loss, model,num_epoch)
#             self.counter = 0

#     def save_checkpoint(self, val_loss, model, num_epoch):
#         '''Saves model when validation loss decrease.'''
#         if self.verbose:

#             self.console.log(f'[red]    => Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).')
#             # print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
#         # torch.save(model.state_dict(), '.\output\model\checkpoint%d.pt' % num_epoch )
#         torch.save(model.state_dict(), self.savepath + '/valid_best_checkpoint.pth')
#         self.val_loss_min = val_loss



 #MCAnet  ACC
# -*- coding:utf-8 -*-

import numpy as np
import torch


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, savepath=None, patience=7, verbose=False, delta=0, num_n_fold=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = -np.inf
        self.early_stop = False
        self.delta = delta
        self.num_n_fold = num_n_fold
        self.savepath = savepath

    def __call__(self, score, model, num_epoch):

        if self.best_score == -np.inf:
            self.save_checkpoint(score, model, num_epoch)
            self.best_score = score

        elif score < self.best_score + self.delta:
            self.counter += 1
            print(
                f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.save_checkpoint(score, model, num_epoch)
            self.best_score = score
            self.counter = 0

    def save_checkpoint(self, score, model, num_epoch):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(
                f'Have a new best checkpoint: ({self.best_score:.6f} --> {score:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.savepath +
                   '/valid_best_checkpoint.pth')
