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
n_channels_convo = 16 #nombre de channels pour la première convolution 3 x 3
n_channels_concat = nb_cluster + n_channels_convo #nombre de channels total lors de la concaténation
n_channels_one_one = 3 #nombre de filtre lors de la conv 1x1

#données
X_tens = torch.randint(150,(200,n_channels_data,24,24))
print("X_tens shape : ",X_tens.shape)


conv4_un_un = torch.nn.Conv2d(n_channels_concat,n_channels_one_one,1)

c = ConstellationNet(nb_cluster,n_channels_data,n_channels_convo,n_channels_concat,n_channels_one_one)

#test de la partie conv
"""
res = c.conv(X_tens.float()) 
X_constell = torch.rand(200,nb_cluster,res.shape[2],res.shape[3]) 
#print("res shape : ",res.shape)


#test de la partie concat
ensemble = c.concatenation(X_constell,res,conv4_un_un)
print("dim de ensemble : ",ensemble.shape)
"""


#conv4
#c.conv4_constell(X_tens.float())







#test de resblock
print("test de resblock")

#convolution

"""
res1conv1 = torch.nn.Conv2d(n_channels_data,64,3,padding = 1)
batch_norm_res1 = torch.nn.BatchNorm2d(64)
conv_un_un_res1 = torch.nn.Conv2d(nb_cluster + 64,n_channels_one_one,1)



c.resblock(X_tens,res1conv1,batch_norm_res1,conv_un_un_res1)
"""


#test de init_convo_layers
#c.init_convo_layers()

#test de resnet12_constell
