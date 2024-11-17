import torch
import hydra
from torch.utils.data import Dataset
from hydra.utils import get_original_cwd

import numpy as np
from utils.LossFunction import CELoss,PolyLoss

def weight_loss(args):

    "set loss function weight"

    if args.ds.data in ['Davis']:
        weight_loss = torch.FloatTensor([0.3, 0.7]).cuda()

    elif args.ds.data in ['KIBA']:
        weight_loss = torch.FloatTensor([0.2, 0.8]).cuda()
       
    elif args.ds.data:
        weight_loss = None
    
    else:
        raise Exception('Unknown weight_loss !')
    
    return weight_loss

def get_loss_fn(args, weight_loss):

    "set loss function weight"

    if args.ds.loss == 'PolyLoss':
        loss_fn = PolyLoss(weight_loss=weight_loss,
                            epsilon=args.ds.loss_epsilon,
                            batch=args.ds.batch_size)
                        
    elif args.ds.loss == 'CrossEntroy':
        loss_fn = CELoss(weight_CE=weight_loss)
    
    elif args.ds.loss == 'CE_DeepDTA':
        loss_fn = CELoss(weight_CE=None)
    
    else:
        raise Exception('Unknown loss function!')
    return loss_fn


def load_data(args):

    dir_input = (get_original_cwd()+'/data/{}.txt'.format(args.ds.data))
    with open(dir_input, "r") as f:
        data_list = f.read().strip().split('\n')

    data_list = shuffle_dataset(data_list, args)
    return data_list


def get_kfold_data(i, datasets, k):

    fold_size = len(datasets) // k 

    val_start = i * fold_size
    if i != k - 1 and i != 0:
        val_end = (i + 1) * fold_size
        validset = datasets[val_start:val_end]
        trainset = datasets[0:val_start] + datasets[val_end:]
    elif i == 0:
        val_end = fold_size
        validset = datasets[val_start:val_end]
        trainset = datasets[val_end:]
    else:
        validset = datasets[val_start:] 
        trainset = datasets[0:val_start]

    return trainset, validset




def shuffle_dataset(dataset, args):
    np.random.seed(args.basic.seed)
    np.random.shuffle(dataset)
    return dataset

def split_data(data_list):

    split_pos = len(data_list) - int(len(data_list) * 0.2)
    train_data_list = data_list[0:split_pos]
    test_data_list = data_list[split_pos:-1]
    len_train = len(train_data_list)
    len_test = len(test_data_list)
    return train_data_list, test_data_list, len_train, len_test

