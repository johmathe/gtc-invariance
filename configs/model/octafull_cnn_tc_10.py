from escnn import nn
from escnn import gspaces
from escnn import group
from torch_tools.config import Config
from gtc.modules import GonR3ConvBlock, FullyConnectedBlock, GTtoT, Ravel, Linear
from gtc.pooling import TCGroupPoolingEfficient
from collections import OrderedDict




"""
CONV 1
"""

conv1 = Config(
    {
        "type": GonR3ConvBlock,
        "params": {"action": gspaces.fullOctaOnR3,
                   "nonlinearity": None, 
                   "n_channels": 24,
                   "kernel_size": 10,
                   "padding": 0,
                   "bias": False 
                  },
    }
)


"""
GROUP POOL
"""

gpool = Config(
    {
        "type": TCGroupPoolingEfficient,
        "params": {
            "group": group.groups.factory.full_octa_group
        },
    }
)


"""
RAVEL
"""

ravel = Config(
    {
        "type": Ravel,
        "params": {},
    }
)


"""
FC1
"""

FC1 = Config(
    {
        "type": FullyConnectedBlock,
        "params": {
            "out_dim": 64 
        }
    }
)



"""
FC2
"""

FC2 = Config(
    {
        "type": FullyConnectedBlock,
        "params": {
            "out_dim": 64
        }
    }
)


"""
FC3
"""

FC3 = Config(
    {
        "type": FullyConnectedBlock,
        "params": {
            "out_dim": 64
        }
    }
)


"""
LINEAR
"""
linear = Config(
    {
        "type": Linear,
        "params": {
            "out_dim": 10
        }
    }
)


"""
MODEL CONFIG
"""

model_config = OrderedDict(
    {
    "conv1": conv1,
    "gpool": gpool,
    "ravel": ravel,
    "FC1": FC1,
    "FC2": FC2,
    "FC3": FC3,
    "linear": linear
    }
)
