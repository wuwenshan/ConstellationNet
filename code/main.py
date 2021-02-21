# -*- coding: utf-8 -*-


import torch

from mini_imagenet_dataset import MiniImageNetData
from constell_net import ConstellationNet
from utils import sample_query_support,training_prototypical


def apprentissage(model,data,labels,Nc,Ns,Nq,nb_episodes,flag):
    
    liste_losses = []
    liste_acc = []
    
    for i in range(nb_episodes):
        print("#################### EPISODE ",i,"###########################")
        loss,acc = training_prototypical(data,labels,Nc,Ns,Nq,model,flag)
        liste_losses.append(loss)
        liste_acc.append(acc)
        
        
    print("Loss : ",loss)
    print("Acc : ",acc)
    print("len de loss : ",len(liste_losses))
    print("len de liste_acc : ",len(liste_acc))
    







if __name__ == "__main__":
    
    #charger les données de MiniImageNet
    
    datasets_path = "..\\datasets\\"
    train_file = datasets_path+"mini-imagenet-cache-train.pkl"
    val_file = datasets_path+"mini-imagenet-cache-val.pkl"
    test_file = datasets_path+"mini-imagenet-cache-test.pkl"

    imgnet_builder = MiniImageNetData(train_file,val_file,test_file)
    donnees,labels = imgnet_builder.getData()
    
    X_train = donnees[0]
    Y_train = labels[0]
    
    X_val = donnees[1]
    Y_val = labels[1]
    
    X_test = donnees[2]
    Y_test = labels[2]
    
    
    #paramètres du modèle
    k = 32
    nb_head = 8
    n_channels_start = X_train.shape[1]
    n_channels_convo = 3
    n_channels_concat = n_channels_convo + k
    n_channels_one_one = 3
    
    #model
    model = ConstellationNet(k,nb_head,n_channels_start,n_channels_convo,n_channels_concat,n_channels_one_one)
    
    #paramètres few-shot
    Nc = 4 #nombre de classes par épisode
    Ns = 5 #nombre d'exemples dans le support par classe
    Nq = 3 #nombre d'exemples query par classe
    nb_episodes = 2
    flag = 0 #0 pour conv4, 1 pour resnet12
    K = len(torch.unique(Y_train))
    print("K : ",K)
    
    #boucle apprentissage
    apprentissage(model,X_train,Y_train,Nc,Ns,Nq,nb_episodes,flag)