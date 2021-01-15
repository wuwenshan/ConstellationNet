# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 20:04:00 2021

@author: Emmanuel
"""

import scipy
import torchvision
import pickle
import torch 
from torch import utils
import torchmeta


datasets_path = "..\\datasets"



def unpickle(file):
    #import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

"""
train_file = "..\\datasets\\cifar-100-python\\train"
train_data = unpickle(train_file)

test_file = "..\\datasets\\cifar-100-python\\test"
test_data = unpickle(test_file)


print("train_data len : ",len(train_data))
print("nombre de classes : ",len(set(train_data[b'fine_labels'])))


print("test_data len : ",len(test_data))
print("nombre de classes : ",len(set(test_data[b'fine_labels'])))


#récupération des données de train

#X_train = torch.from_numpy(train_data[b'data']).reshape(50000,32,32,32)
#Y_train = torch.tensor(train_data[b'fine_labels'])


#print("shape de X_train : ",X_train.shape)
#print("shape de Y_train : ",Y_train.shape)
"""




#cifar100 (pas utilisé)
"""
c100_dataset_train = torchvision.datasets.CIFAR100("..\\datasets",download = True)
X_train_fc = torch.from_numpy(c100_dataset_train.data)
Y_train_fc = torch.tensor(c100_dataset_train.targets)
print("X_train shape  :",X_train_fc.shape)
print("Y_train shape : ",Y_train_fc.shape)
print("Nombre de classes :",len(torch.bincount(Y_train_fc)))
"""



"""


#fc100_train = torchmeta.datasets.FC100(datasets_path,meta_train = True,download = True)


mini_imagenet_train = torchmeta.datasets.MiniImagenet(datasets_path,64,meta_train = True,download = True)
train_dataset = mini_imagenet_train.dataset

mini_imagenet_val = torchmeta.datasets.MiniImagenet(datasets_path,16,meta_val = True,download = True)
val_dataset = mini_imagenet_val.dataset

mini_imagenet_test = torchmeta.datasets.MiniImagenet(datasets_path,20,meta_test= True,download = True)
test_dataset = mini_imagenet_test.dataset
"""

#print("train_dataset len : ",train_dataset.__len__())







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
            labels.append(int(i/600))
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
   


#MiniImageNet train avec Dataset
train_mini_dataset = MiniImageNetDataset(datasets_path+"\\mini-imagenet-cache-train.pkl")
print("train_mini_dataset size : ",train_mini_dataset.__len__())
print()




#MiniImageNet val avec Dataset
val_mini_dataset = MiniImageNetDataset(datasets_path+"\\mini-imagenet-cache-val.pkl")
print("val_mini_dataset size : ",val_mini_dataset.__len__())
print()


#MiniImageNet test avec Dataset
test_mini_dataset = MiniImageNetDataset(datasets_path+"\\mini-imagenet-cache-test.pkl")
print("test_mini_dataset size : ",test_mini_dataset.__len__())
print()


b_size = 1024
in_train_loader,in_val_loader,in_test_loader = getMiniLoaders(train_mini_dataset,val_mini_dataset,test_mini_dataset,b_size)






