import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as nn
from model.dsbn import DomainSpecificBatchNorm2d
from model.layers import *

def normalization(planes, norm='gn', num_domains=None, momentum=0.1):
    if norm == 'dsbn':
        m = DomainSpecificBatchNorm2d(planes, num_domains=num_domains, momentum=momentum)
    elif norm == 'bn':
        m = nn.BatchNorm2d(planes)
    elif norm == 'gn':
        m = nn.GroupNorm(1, planes)
    elif norm == 'in':
        m = nn.InstanceNorm2d(planes)
    else:
        raise ValueError('Normalization type {} is not supporter'.format(norm))
    return m

class TwoConv(nn.Module):
    def __init__(self, inplace, planes, norm='bn', first=True, num_domains=None, momentum=0.1):
        super(TwoConv, self).__init__()

        

class Unet3plus(nn.Module):
    def __init__(self, c=1, n=32, norm='bn', num_classes=2, num_domains=4, momentum=0.1):
        super(Unet3plus, self).__init__()

