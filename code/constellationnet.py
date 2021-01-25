# -*- coding: utf-8 -*-
import torch
from scipy.spatial import distance_matrix

class ConstellationNet(torch.nn.Module):
    
    def __init__(self,k,n_channels_start,n_channels_convo,n_channels_concat,n_channels_one_one):
        super(ConstellationNet, self).__init__()
        
        self.nb_cluster = k
        
        #param de conv4
        # channels en entrée (RGB), 16 filtres, filtre 3* 3
        self.conv_trois_trois = torch.nn.Conv2d(n_channels_start,n_channels_convo,3)
        
        #conv 1x1 finale
        self.conv_un_un = torch.nn.Conv2d(n_channels_concat,n_channels_one_one,1)
        
        self.relu = torch.nn.ReLU()
        self.max_pool = torch.nn.MaxPool2d(2)
        
        
        #param du resnet
        #residual bloc 1
        self.res1conv1 = torch.nn.Conv2d(n_channels_start,64,3)
        self.res1conv23 = torch.nn.Conv2d(64,64,3)
        self.batch_norm_res1 = torch.nn.BatchNorm2d(64)
        
        #residual block 2
        self.res2conv1 = torch.nn.Conv2d(64,128,3)
        self.res2conv23 = torch.nn.Conv2d(128,128,3)
        self.batch_norm_res2 = torch.nn.BatchNorm2d(128)
        
        #residual block 3
        self.res3conv1 = torch.nn.Conv2d(128,256,3)
        self.res3conv23 = torch.nn.Conv2d(128,256,3)
        self.batch_norm_res3 = torch.nn.BatchNorm2d(256)
        
        #residual block 4
        self.res4conv1 = torch.nn.Conv2d(256,512,3)
        self.res4conv23 = torch.nn.Conv2d(512,512,3)
        self.batch_norm_res4 = torch.nn.BatchNorm2d(512)
        
        
        
    def forward(self, x):
        pass
    
    def conv(self,X):
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
        
    
    
    def concatenation(self,X_constell,X_feature_map):
        
        #concaténation entre la Convolutional Feature Map et la sortie du Constellation Network
        concat = torch.cat((X_constell,X_feature_map),1)
        
        #conv 1x1
        conv_one_one = self.conv_un_un(concat)
        
        #BatchNorm
        concat_batch_norm = torch.nn.BatchNorm2d(conv_one_one.shape[1])(conv_one_one)
        
        #Relu
        relu_concat = self.relu(concat_batch_norm)
        
        return relu_concat
        
    
    
    def resblock(self,X,X_constell,convo1,convo23,batch_norm):
        
        #première convo du bloc
        X_convo = convo1(X)
        X_b_norm = torch.nn.BatchNorm2d(X_convo.shape[1])(X_convo)
        X_feature_map = self.relu(X_b_norm)
        
        #identité
        ident = X_convo
        
        #Constellation network 
        X_constell = self.constellation(X_feature_map)
        
        #concat Feature map et Constellation
        X_concat = self.concatenation(X_constell,X_feature_map)
        
        #2e et 3e convo du bloc
        for i in range(2):
            X_convo_i = convo23(X_concat)
            X_convo_i_norm = torch.nn.BatchNorm2d(X_convo_i.shape[1])(X_convo_i)
            X_convo_i_feature_map = self.relu(X_convo_i_norm)
            
            #constellation et concaténation
            X_i_constell = self.constellation(X_convo_i_feature_map)
            X_concat = self.concatenation(X_i_constell,X_convo_ifeature_map)
            
        
        #max pool
        X_end_block = self.max_pool(X_concat)
        
        #ajouter idendité
        #ajouter paddind et peut être stride pour garder les mêmes dims
      
    
   
    
    
    
    
        
    
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
