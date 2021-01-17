# -*- coding: utf-8 -*-

import torch
from scipy.spatial import distance_matrix

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
        
    
    
    def concatenation(self,X_constell,X_feature_map):
        
        #concaténation entre la Convolutional Feature Map et la sortie du Constellation Network
        concat = torch.cat((X_constell,X_feature_map),1)
        
        #conv 1x1
        conv_one_one = self.conv3(concat)
        
        #BatchNorm
        concat_batch_norm = torch.nn.BatchNorm2d(conv_one_one.shape[1])(conv_one_one)
        
        #Relu
        relu_concat = self.relu(concat_batch_norm)
        
        return relu_concat
        

        
    
    def constell(self):
        pass
    
    
    
    
    """
        Cette étape de clustering se compose d'un soft k-means produisant une "distance map"
        Indication du papier :
            - beta > 0
            - lambda = 1.0
    """
    
    
    def cellFeatureClustering(cellFeature, k, epoch, beta=100, lbda=1.0):
    
        batch_size = cellFeature.shape[0]
        h_size = cellFeature.shape[1]
        
        # initialisation
        s = torch.zeros(k)
        v = torch.randn(k, cellFeature.shape[3])
        cellFeature = cellFeature.view(-1, cellFeature.shape[3])
        
        for _ in range(epoch):
        
            # cluster assignment
            d = torch.tensor(distance_matrix(cellFeature, v))
            
            m = torch.zeros(d.shape)
            for i in range(len(m)):
                m[i] = torch.exp(-beta * d[i]) / torch.sum( torch.exp(-beta * d[i]) )
                
            v_p = torch.zeros(v.shape)
            for i in range(len(v_p)):
                v_p[i] = torch.sum(m[:,i]) * torch.sum(cellFeature[i]) / torch.sum(m[:,i])
            
            # centroid movement
            delta_s = torch.sum(m, 0)
            mu = lbda / (s + delta_s)
            v = (1 - mu.unsqueeze(1)) * v + mu.unsqueeze(1) * v_p
            
            # counter update
            s += delta_s
        
        d_map = d.view(batch_size, h_size, h_size, d.shape[1])
    
        return d_map
