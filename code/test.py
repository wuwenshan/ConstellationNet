# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 20:21:39 2021

@author: Emmanuel
"""

import torch
import numpy as np
from constellationnet import ConstellationNet
from conv4_constell import Conv4Constell
from resnet12_constell import ResNet12Constell
import constell_net

from mini_imagenet_dataset import MiniImageNetData
#from cifar_dataset import get_CIFARFS
from utils import sample_query_support,training_prototypical,sim_acc




#channels number
nb_cluster = 32 #nombre de chanels pour le Feature Cell Encoding
nb_head = 8
n_channels_data = 3 
n_channels_convo = 64 #nombre de channels pour la première convolution 3 x 3
n_channels_concat = nb_cluster + n_channels_convo #nombre de channels total lors de la concaténation
n_channels_one_one = 3 #nombre de filtre lors de la conv 1x1

#données


X_tens = torch.randn((30,n_channels_data,32,32)).float()

#print("X_tens shape : ",X_tens.shape)


conv4_un_un = torch.nn.Conv2d(n_channels_concat,n_channels_one_one,1)

c = ConstellationNet(nb_cluster,nb_head,n_channels_data,n_channels_convo,n_channels_concat,n_channels_one_one)



#test de la partie conv
res = c.conv(X_tens.float()) 
#X_constell = torch.rand(200,nb_cluster,res.shape[2],res.shape[3]) 


"""
#print("res shape : ",res.shape)
"""

"""
#test de la partie concat
ensemble = c.concatenation(X_constell,res,conv4_un_un)
print("dim de ensemble : ",ensemble.shape)
"""


#test de constell
#c.constell(res) #erreur

#conv4
#c.conv4_constell(X_tens.float())

#conv4 sequential
#c.conv4(X_tens)


#forward
#c(X_tens)

"""
output = c(X_tens)
print("shape output : ",output.shape)
"""








#test de resblock
#print("test de resblock")

#convolution

"""
res1conv1 = torch.nn.Conv2d(n_channels_data,64,3,padding = 1)
batch_norm_res1 = torch.nn.BatchNorm2d(64)
conv_un_un_res1 = torch.nn.Conv2d(nb_cluster + 64,n_channels_one_one,1)
"""


#c.resblock(X_tens,res1conv1,batch_norm_res1,conv_un_un_res1)



#test de init_convo_layers
#c.init_convo_layers()

#test de resnet12_constell
#c.resnet12_constell(X_tens.double())





###############################

#test de Conv4Constell

"""
conv4 = Conv4Constell(n_channels_data,n_channels_convo,c)
conv4(X_tens)
"""

#test de Resnet12Constell
#resnet12 = ResNet12Constell(n_channels_data,c)
#resnet12(X_tens)

#######################

#test de ConstellationNet dans le fichier conv4_constell_proto
model = constell_net.ConstellationNet(nb_cluster,nb_head,n_channels_data,n_channels_convo,n_channels_concat,n_channels_one_one)

""" Partie conv4 """
#model(X_tens)

""" Partie resnet12 """
#model.resnet12_constell(X_tens)

""" Assemblage des deux modèles """

#archi = 0
#model(X_tens,archi)

###################################
#MiniImageNetDataset
datasets_path = "..\\datasets\\"
train_file = datasets_path+"mini-imagenet-cache-train.pkl"
val_file = datasets_path+"mini-imagenet-cache-val.pkl"
test_file = datasets_path+"mini-imagenet-cache-test.pkl"

imgnet_builder = MiniImageNetData(train_file,val_file,test_file)


#test de create

debut_classe = 0
X_train,Y_train,data = imgnet_builder.create(train_file,debut_classe)
print("Len de Y_train : ",Y_train.shape)

"""
ind_classe = 6227
class_dict = data['class_dict']
print("Classe de Y indice",ind_classe," : ",Y_train[ind_classe])
if ind_classe in list(class_dict.values())[Y_train[ind_classe] - debut_classe]:
    print("Bonne classe")
    print("Classe associée : ",Y_train[ind_classe])
else :
    print("Mauvaise classe")
    print("Y_train[30000] = ",Y_train[ind_classe])
    print("class dict associé : ",list(class_dict.values())[Y_train[ind_classe]])
    print("Y_train : ",Y_train)
"""


#test de getData
donnes,labels = imgnet_builder.getData()

#sample_query_support

mini_x = X_train[:10]
mini_y = Y_train[:10]

y_tens = torch.tensor([0]*10 + [1] *10 + [2]*10)
print("y_tens : ",y_tens)
nb_cluster = 32
nb_head = 1
n_channels_data = X_train.shape[1] 
n_channels_convo = 64
n_channels_concat = nb_cluster + n_channels_convo
n_channels_one_one = 3

network = constell_net.ConstellationNet(nb_cluster,nb_head,n_channels_data,n_channels_convo,n_channels_concat,n_channels_one_one)

X_tens = X_train
y_tens = Y_train

ns = 5
nq = 2
K = len(torch.unique(y_tens))
Nc = 2
flag = 0
k = 3
#sample_query_support(X_train,Y_train,ns,nq,k)


#training_prototypical
loss,acc = training_prototypical(X_tens,y_tens,Nc,ns,nq,network,flag)


print("loss : ",loss)
print("acc : ",acc)


input1 = torch.randn(100, 128)
input2 = torch.randn(100, 128)
cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
output = cos(input1, input2)
#print("output shape : ",output.shape)

#sim_acc
"""
d = torch.tensor([[0.4,0.5,0.3,],
                  [0.25,0.3,0.1],
                  [0.18,0.99,0.95],
                  [0.15,0.37,0.28]
                    ])

y = torch.tensor([0,4,2,0])

V = np.array([0,2,4])

acc = sim_acc(d,y,V)
print(f"accuracy : {acc}")
"""
