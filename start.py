#!/usr/bin/env python
# coding=utf-8
'''
Author: Zhan
Date: 2023-08-01 00:38:22
'''

import os
import sys
import hydra
from hydra.utils import get_original_cwd, to_absolute_path
from pathlib import Path


@hydra.main(version_base=None, config_path='conf', config_name = 'start')
def start(config):

    print(config)
    os.system("echo 'start Training!!!'")

    res_save = Path(config.save)
    Path(res_save).mkdir(parents = True, exist_ok = True)

    cmd = f"""cd {get_original_cwd()}&&CUDA_VISIBLE_DEVICES={config.gpu} python main.py \
        ds={config.data} \
        ds.r_savepath={res_save}  \
        ds.d_path={config.outpath} \
        ds.Epoch={config.epoch} \
        ds.loss={config.loss} \
        ds.c_p={config.c_p} \
        ds.c_d={config.c_d} \
        ds.batch_size={config.batch_size}
        """
    print(cmd)
    os.system(cmd)
    

if __name__ == '__main__':
    
    start()