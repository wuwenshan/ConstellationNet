# -*- coding: utf-8 -*-

import torch

class ConstellationNet(torch.nn.Module):
    
    def __init__(self,n_channels_int,n_channels_end,n_channels_one_one):
        super(ConstellationNet, self).__init__()
        
        # 3 channels en entrée (RGB), 16 filtres, filtre 3* 3
        self.conv1 = torch.nn.Conv2d(3,16,3)
        
        #pour les convolutions intérieurs 
        self.conv2 = torch.nn.Conv2d(n_channels_int,16,3)
        
        #conv 1x1 finale
        self.conv3 = torch.nn.Conv2d(n_channels_end,n_channels_one_one,1)
        
        self.relu = torch.nn.ReLU()
        self.max_pool = torch.nn.MaxPool2d(2)
        
        
        
        
    def forward(self, x):
        pass
    
    def conv(self,X):
        #convolution 3 * 3
        convo = self.conv1(X)
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
        
    
    
    
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    def constell(self):
        pass