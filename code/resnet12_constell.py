# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 18:10:42 2021

@author: Emmanuel
"""


import torch

class ResNet12Constell(torch.nn.Module):
    def __init__(self, n_channels_start,constell_net):
        super().__init__()
        
        self.relu = torch.nn.ReLU()
        self.max_pool = torch.nn.MaxPool2d(2)
        
        self.constell_net = constell_net
        
        self.init_convo_layers()
        

    def init_convo_layers(self):
        """ initialiser les layers de convolutions et de normalisation pour les différents blocs résiduels """
        
        #convo 3 x 3  avec n_channels = 64,128,256,512
        convo_trois_trois_layers = []
        convo_one_one_layers = []
        batch_norm_layers = []
        
        i = 1
        n_channels_res_convo = 64
        
        convo_trois_trois_layers.append(torch.nn.Conv2d(self.constell_net.n_channels_start,n_channels_res_convo,3,padding = 1))
        convo_one_one_layers.append(torch.nn.Conv2d(self.constell_net.nb_cluster + n_channels_res_convo,self.constell_net.n_channels_one_one,1))
        batch_norm_layers.append(torch.nn.BatchNorm2d(n_channels_res_convo))
        
        while i < 4:
            
            #nombre de filtres doublés d'un bloc à l'autre
            n_channels_res_convo *= 2
            convo_trois_trois_layers.append(torch.nn.Conv2d(self.constell_net.n_channels_start,n_channels_res_convo,3,padding = 1))
            convo_one_one_layers.append(torch.nn.Conv2d(self.constell_net.nb_cluster + n_channels_res_convo,self.constell_net.n_channels_one_one,1))
            batch_norm_layers.append(torch.nn.BatchNorm2d(n_channels_res_convo))
            
            i += 1
            
        
        
        self.convo_trois_trois_layers = convo_trois_trois_layers
        self.convo_one_one_layers = convo_one_one_layers
        self.batch_norm_layers = batch_norm_layers


    def resblock(self,X,convo1,batch_norm,conv1):
        """ residual block de ResNet-12 """
        
        #print("dim de X : ",X.shape)
        
        #première convo du bloc
        X_convo = convo1(X.float())
        X_b_norm = torch.nn.BatchNorm2d(X_convo.shape[1])(X_convo)
        X_feature_map = self.relu(X_b_norm)
        
        #identité
        ident = X
        #print("dim de ident : ",ident.shape)
        
        #Constellation network 
        X_constell = torch.randn(X.shape[0],self.constell_net.nb_cluster,X.shape[2],X.shape[3])
        #print("X_constell shape : ",X_constell.shape)
        
        #concat Feature map et Constellation
        
        X_concat = self.constell_net.concatenation(X_constell,X_feature_map,conv1)
        #print("X_concat shape : ",X_concat.shape)
                                      
        
        #2e et 3e convo du bloc
        for i in range(2):
            #print(i+2,"e convo du bloc")
            X_convo_i = convo1(X_concat)
            X_convo_i_norm = torch.nn.BatchNorm2d(X_convo_i.shape[1])(X_convo_i)
            X_convo_i_feature_map = self.relu(X_convo_i_norm)
            #print("shape x convo i : ",X_convo_i_feature_map.shape)
            
            
            #constellation et concaténation
            #X_i_constell = self.constellation(X_convo_i_feature_map)
            X_i_constell = torch.randn(X_convo_i.shape[0],self.constell_net.nb_cluster,X_convo_i.shape[2],X_convo_i.shape[3])
            X_concat = self.constell_net.concatenation(X_i_constell,X_convo_i_feature_map,conv1)
            #print("resbloc, shape de x concat : ",X_concat.shape)
        
        
        
        return X_concat
    
    
    
    
    
    
    
    def forward(self,X):
        """ ResNet-12 backbone avec module Constell """
        
        i = 0
        
        X_ident = X
        
        for i in range(4):
            
            #bloc résiduel 
            X_res = self.resblock(X_ident,self.convo_trois_trois_layers[i],self.batch_norm_layers[i],self.convo_one_one_layers[i])
            
            #ajouter identité
            X_out = X_ident + X_res
            
            #max pool (identité pour le prochain bloc)
            X_ident = self.max_pool(X_out)
        
            print("num_bloc : ",i+1,"X_ident shape: ",X_ident.shape)
        
        
        return X_ident
    
    
    
    