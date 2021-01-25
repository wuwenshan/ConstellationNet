# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 20:21:39 2021

@author: Emmanuel
"""

import torch

from constellationnet import ConstellationNet


#channels number
nb_cluster = 5 #nombre de chanels pour le Feature Cell Encoding
n_channels_data = 3 
n_channels_convo = 16
n_channels_concat = nb_cluster + 16 #nombre de channels total lors de la concaténation
n_channels_one_one = 3 #nombre de filtre lors de la conv 1x1

#données
X_tens = torch.randint(150,(200,n_channels_data,24,24))
print("X_ten shape : ",X_tens.shape)




c = ConstellationNet(nb_cluster,n_channels_data,n_channels_convo,n_channels_concat,n_channels_one_one)

#test de la partie conv
res = c.conv(X_tens.float()) 
print("res shape : ",res.shape)


#test de la partie concat
X_constell = torch.rand(200,nb_cluster,res.shape[2],res.shape[3]) 
ensemble = c.concatenation(X_constell,res)
print("dim de ensemble : ",ensemble.shape)