# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 20:21:39 2021

@author: Emmanuel
"""

import torch

from constellationnet import ConstellationNet


X_tens = torch.randint(150,(200,3,12,12))

n_chanels_int = 10


c = ConstellationNet(n_chanels_int)
res = c.conv(X_tens.float())