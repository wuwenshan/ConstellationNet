# -*- coding: utf-8 -*-


import scipy
import torchvision
import pickle
import torch 
from torch import utils
#import torchmeta


datasets_path = "..\\datasets"










class MiniImageNetDataset(utils.data.Dataset):
    def __init__(self,path):
        self.path = path
        self.X,self.Y = self.create()
    
    
    def unpickle(self,file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict
    
    
    
    def getLabels(self,X):
        labels = []
        for i in range(len(X)):
            labels.append(i//600)
        return labels
    
    
    def create(self):
        data_imagenet = self.unpickle(self.path)
        X_imagenet = torch.from_numpy(data_imagenet['image_data'])
        print("X : ",X_imagenet.shape)
        
        
        Y_imagenet = torch.tensor(self.getLabels(X_imagenet))
        #print("Y_train shape : ",Y_train_imagenet.shape)
        
        
        #assert(Y_train_imagenet[603] == 1),print("Y_train 603 : ",Y_train_imagenet[603])
        
        print("Nombre de chaque classe : ",torch.bincount(Y_imagenet))
        print("Nombre de classes :",len(torch.bincount(Y_imagenet)))
        
        
        return X_imagenet,Y_imagenet
        
    
    
    def __getitem__(self,i):
        return self.X[i],self.Y[i]
    
    
    def __len__(self):
        return self.X.shape[0]







 
def getMiniLoaders(train_ds,val_ds,test_ds,batch_size):
    
    
    mini_train_loader = utils.data.DataLoader(train_ds,batch_size,shuffle = True)
    mini_val_loader = utils.data.DataLoader(val_ds,batch_size,shuffle = True)
    mini_test_loader = utils.data.DataLoader(test_ds,batch_size,shuffle = True)
    return mini_train_loader,mini_val_loader,mini_test_loader
   


def getMiniDatasets(path):
    #MiniImageNet train avec Dataset
    train_mini_dataset = MiniImageNetDataset(path+"\\mini-imagenet-cache-train.pkl")
    print("train_mini_dataset size : ",train_mini_dataset.__len__())
    print()




    #MiniImageNet val avec Dataset
    val_mini_dataset = MiniImageNetDataset(path+"\\mini-imagenet-cache-val.pkl")
    print("val_mini_dataset size : ",val_mini_dataset.__len__())
    print()


    #MiniImageNet test avec Dataset
    test_mini_dataset = MiniImageNetDataset(path+"\\mini-imagenet-cache-test.pkl")
    print("test_mini_dataset size : ",test_mini_dataset.__len__())
    print()

    return train_mini_dataset,val_mini_dataset,test_mini_dataset

"""
b_size = 1024
in_train_loader,in_val_loader,in_test_loader = getMiniLoaders(train_mini_dataset,val_mini_dataset,test_mini_dataset,b_size)
"""

class MiniImageNetData():
    def __init__(self,train_path,val_path,test_path):
        self.files = [train_path,val_path,test_path]
        
        
    def unpickle(self,file):
            with open(file, 'rb') as fo:
                dict = pickle.load(fo, encoding='bytes')
            return dict
        
        
        
    def getLabels(self,X,start_class):
        labels = []
        for i in range(len(X)):
            labels.append(start_class + i//600)
        return labels
        
        
    def create(self,path,start_class):
        data_imagenet = self.unpickle(path)
        X_imagenet = torch.from_numpy(data_imagenet['image_data'])
        print("X_imagenet avant view : ",X_imagenet.shape)
        X = X_imagenet.view(X_imagenet.shape[0],X_imagenet.shape[3],
                            X_imagenet.shape[1],X_imagenet.shape[2])
        
        print("X après view : ",X.shape)
        
        
        
        
        
        Y_imagenet = torch.tensor(self.getLabels(X_imagenet,start_class))
        #print("Y_train shape : ",Y_train_imagenet.shape)
        
        
        #assert(Y_train_imagenet[603] == 1),print("Y_train 603 : ",Y_train_imagenet[603])
        print("Différentes classes : ",torch.unique(Y_imagenet))
        print("Nombre de classes différents : ",len(torch.unique(Y_imagenet)))
        print()
        
        
        return X,Y_imagenet,data_imagenet
        
    def getData(self):
        donnees = []
        labels = []
        start_class = 0
        for i in range(len((self.files))):
            X,Y,data = self.create(self.files[i],start_class)
            donnees.append(X)
            labels.append(Y)
            start_class = torch.max(Y) + 1
            
        return donnees,labels
