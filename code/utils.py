# -*- coding: utf-8 -*-

import torch
import cifar_dataset as cf
import numpy as np

def unpickle(file): 
    import pickle 
    with open(file, 'rb') as fo: 
        dict = pickle.load(fo, encoding='bytes') 
    return dict


def getSupportQuery(data, label, K=3, N=5):
    cl = np.random.choice(label, K, replace=False)
    
    print("cl : ", cl)
    
    support_data = []
    support_label = []
    query_data = []
    query_label = []
    
    for c in cl:
        data_c = data[torch.where(label == c)[0]]
        print("len : ", len(data_c))
        idx = torch.randperm(len(data_c))[:N]
        print("idx : ", idx)
        print("1 : ", data_c[idx].shape)
        print("2 : ", data_c.shape)
        support_data.append(data_c[idx])
        support_label.append(torch.tensor([c]*N))
        query_data.append(data_c)
        query_label.append(torch.tensor([c]*len(data_c)))
        
    return torch.cat(support_data), torch.cat(support_label), torch.cat(query_data), torch.cat(query_label)


train_data, train_label, test_data, test_label, val_data, val_label = cf.get_CIFARFS(train_size=64, test_size=20, val_size=16, batch_size=100, dataset="cifarfs")

print(train_data.shape, train_label.shape)

supp_data, supp_label, query_data, query_label = getSupportQuery(train_data, train_label)

print(supp_data.shape, supp_label.shape, query_data.shape, query_label.shape)