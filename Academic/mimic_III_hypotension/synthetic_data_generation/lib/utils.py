# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  Copyright (c) 2021. by Sebastiano Barbieri, UNSW.                                     +
#  All rights reserved. This file is part of the Health Gym, and is released under the   +
#  "MIT License Agreement". Please see the LICENSE file that should have been included   +
#  as part of this package.                                                              +
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import math
import pickle as pkl
from pdb import set_trace as bp

import numpy as np
import pandas as pd
import torch


def save_obj(obj, name):
    with open(name, "wb") as f:
        pkl.dump(obj, f, pkl.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name, "rb") as f:
        return pkl.load(f)


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")
    return device


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
