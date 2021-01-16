# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 20:21:39 2021

@author: Emmanuel
"""

import torch

from constellationnet import ConstellationNet

nb_cluster = 5 #nombre de chanels pour le Feature Cell Encoding
X_tens = torch.randint(150,(200,3,12,12))

n_chanels_int = 10 #nombre de channels en entrée pour les blocks 2,3 et 4
n_channels_concat = nb_cluster + 16 #nombre de channels total lors de la concaténation
n_channels_one_one = 1 #nombre de filtre lors de la conv 1x1

c = ConstellationNet(n_chanels_int,n_channels_concat,n_channels_one_one)

#test de la partie conv
res = c.conv(X_tens.float()) 
print("res shape : ",res.shape)

#test de la partie concat
X_constell = torch.rand(200,nb_cluster,res.shape[2],res.shape[3]) 
ensemble = c.concatenation(X_constell,res)
print("dim de ensemble : ",ensemble.shape)