
import torch
import numpy as np
import random
import os


def set_seed(seed=0, cudnn='normal'):
    '''
    [ 'benchmark', 'normal', 'slow', 'none' ] from left to right, cudnn randomness decreases, speed decreases
    'benchmark': turn on CUDNN_FIND to find the fast operation, when each iteration has the same computing graph (e.g. input size and model architecture), it can speed up a bit
    'normal': usually used option, accuracy differs from the digit on 0.1%
    'slow': it slows down computation. More accurate reproducing than 'normal', especially when gpu number keep unchanged, the accuracy is almost the same.
    'none'：running on gpu/cpu yields the same result, but is slow.
    '''


    assert type(seed) == int and seed in range(0,4294967296), "`seed` must be anint in [0,4294967295]"
    assert cudnn in [ 'benchmark', 'normal', 'none', 'slow' ], "`cudnn` must be in [ 'benchmark', 'normal', 'slow', 'none' ] "

    os.environ['PYTHONHASHSEED'] = str(seed) # seed for hash() function, affects the iteration order of dicts, sets and other mappings, str(seed) int [0; 4294967295]
    random.seed(seed) # random and transforms, seed int or float
    np.random.seed(seed) #numpy, seed int
    torch.manual_seed(seed) # cpu, seed int or float
    torch.cuda.manual_seed(seed) # gpu, seed int or float
    torch.cuda.manual_seed_all(seed) # multi-gpu, seed int or float

    if cudnn=='none':
        torch.backends.cudnn.enabled = False # if True, cudnn accelarate, similar result but not exactly same
    elif cudnn=='slow':
        # when cuDNN is using deterministic mode, computing may be slown (depends on the model)
        # low affect on reproducing, only changes digits after the decimal point. not recommended to use unless requires exact reproducing。
        torch.backends.cudnn.deterministic = True  # if True, cudnn has no randomness，cpu/gpu yield same result, but slow down convolution
        torch.backends.cudnn.benchmark = False   # if True, turn on CUDNN_FIND to find the fast operation
    elif cudnn == 'normal':
        torch.backends.cudnn.benchmark = False   # if True, turn on CUDNN_FIND to find the fast operation
    elif cudnn=='benchmark':
        torch.backends.cudnn.benchmark = True    # if True, turn on CUDNN_FIND to find the fast operation


def set_work_init_fn(seed):
    def worker_init_fn(worker_id):
        np.random.seed(seed  + worker_id)
    return worker_init_fn
