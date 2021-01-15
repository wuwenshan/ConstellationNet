# -*- coding: utf-8 -*-

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np

class DatasetCIFARFS(Dataset):
    
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return len(self.y)

def get_CIFARFS(train_size=64, test_size=20, val_size=16, batch_size=100):
    
    if train_size + test_size + val_size == 100:
    
        transform = transforms.Compose([transforms.ToTensor()])
        
        train_dataset = torchvision.datasets.CIFAR100(root='../data', train=True, transform=transform, download=True)
        test_dataset = torchvision.datasets.CIFAR100(root='../data', train=False, transform=transform, download=True)
        dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])
        
        cl = torch.randperm(100)
        train_cl = cl[:train_size]
        test_cl = cl[train_size: train_size+test_size]
        val_cl = cl[train_size+test_size:]
        
        train_data = []
        train_label = []
        
        test_data = []
        test_label = []
        
        val_data = []
        val_label = []
        
        for data, label in dataset:
            if label in train_cl:
                train_data.append(data)
                train_label.append(label)
            elif label in test_cl:
                test_data.append(data)
                test_label.append(label)
            elif label in val_cl:
                val_data.append(data)
                val_label.append(label)
            else:
                print("wtf : ", label)
        
        
        train_ds = DatasetCIFARFS(train_data, train_label)
        test_ds = DatasetCIFARFS(test_data, test_label)
        val_ds = DatasetCIFARFS(val_data, val_label)
        
        train_dl = DataLoader(train, batch_size=TRAIN_BATCHSIZE, collate_fn=TextDataset.collate)
        
    else:
        print("Train, Test et Val doivent etre egales a 100")
        return None
    
    return None


get_CIFARFS()