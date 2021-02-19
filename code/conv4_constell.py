# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 17:26:15 2021

@author: Emmanuel
"""
import torch

class Conv4Constell(torch.nn.Module):
    def __init__(self, n_channels_start, n_channels_convo,constell_net):
        super().__init__()
        
        #param de conv4
        
        # channels en entrée (RGB), 16 filtres, filtre 3* 3
        self.conv_trois_trois = torch.nn.Conv2d(n_channels_start,n_channels_convo,3,padding = 1)
        self.relu = torch.nn.ReLU()
        self.max_pool = torch.nn.MaxPool2d(2)
        #self.conv_un_un = torch.nn.Conv2d(n_channels_concat,n_channels_one_one,1)

        self.constell_net = constell_net
        
    
    def conv(self,X):
        """ bloc simple de convolution pour Conv-4 """
        
        #convolution 3 * 3
        convo = self.conv_trois_trois(X)
        #print("Après la convo shape de X : ",convo.shape)
        
        #batch_norm
        batch_norm_convo = torch.nn.BatchNorm2d(convo.shape[1])(convo)
        #print("Après normalisation : ",batch_norm_convo.shape)
        
        #relu
        relu_convo = self.relu(batch_norm_convo)
        #print("Après relu : ",relu_convo.shape)
        
        #max pool
        convo_feature_map = self.max_pool(relu_convo)
        #print("Après max pool  : ",convo_feature_map.shape)
        
        return convo_feature_map
    
    
    
    def forward(self,X):
        for _ in range(4):
            # convolutionnal feature map
            cfm = self.conv(X)
            
            # cell relation modeling
            constell_module = self.constell_net.constell(cfm)
            
            x = self.constell_net.concatenation(cfm, constell_module, self.constell_net.conv_un_un)
            
    
        return x # self.constell_net.similarity(x)