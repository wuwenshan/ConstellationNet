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
    
    support_data = []
    support_label = []
    query_data = []
    query_label = []
    
    for c in cl:
        data_c = data[torch.where(label == c)[0]]
        idx = torch.randperm(len(data_c))[:N]
        support_data.append(data_c[idx])
        support_label.append(torch.tensor([c]*N))
        query_data.append(data_c)
        query_label.append(torch.tensor([c]*len(data_c)))
        
    return torch.cat(support_data), torch.cat(support_label), torch.cat(query_data), torch.cat(query_label)
