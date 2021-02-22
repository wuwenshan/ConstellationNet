# -*- coding: utf-8 -*-

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

class DatasetCIFAR(Dataset):
    
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return len(self.y)
    
    
def unpickle(file): 
    import pickle 
    with open(file, 'rb') as fo: 
        dict = pickle.load(fo, encoding='bytes') 
    return dict
    

def get_CIFARFS(train_size=64, test_size=20, val_size=16, dataset="cifarfs"):
    
    
    if dataset == "cifarfs":
        
        if train_size + test_size + val_size != 100:
            print("Train, Test et Val doivent sommer à 100")
            return None
        
        else:
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
                data = data.permute(1, 2, 0)
                #data = data.view(-1).unsqueeze(0) # (3, 32, 32) -> (1, 3x32x32)
                if label in train_cl:
                    train_data.append(data)
                    train_label.append(torch.tensor([label]))
                elif label in test_cl:
                    test_data.append(data)
                    test_label.append(torch.tensor([label]))
                elif label in val_cl:
                    val_data.append(data)
                    val_label.append(torch.tensor([label]))
                else:
                    print("wtf : ", label)
                    
                    
            return torch.stack(train_data).permute(0, 3, 1, 2)/255, torch.cat(train_label), torch.stack(test_data).permute(0, 3, 1, 2)/255, torch.cat(test_label), torch.stack(val_data), torch.cat(val_label)
        
            
            
            """
            
            train_ds = DatasetCIFAR(train_data, train_label)
            test_ds = DatasetCIFAR(test_data, test_label)
            val_ds = DatasetCIFAR(val_data, val_label)
            
            train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
            test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=True)
            val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=True)
            
            return train_dl, test_dl, val_dl
        
            """
            
    elif dataset == "fc100":
        
        if train_size + test_size + val_size != 20:
            print("Train, Test et Val doivent sommer à 20")
            return None
        
        else:
            train_dataset = unpickle("../data/cifar-100-python/train")[b"data"]
            test_dataset = unpickle("../data/cifar-100-python/test")[b"data"]
            train_coarse_labels = unpickle("../data/cifar-100-python/train")[b"coarse_labels"]
            test_coarse_labels = unpickle("../data/cifar-100-python/test")[b"coarse_labels"]
            
            coarse_labels = torch.cat((torch.tensor(train_coarse_labels), torch.tensor(test_coarse_labels)), 0)
            dataset = torch.cat((torch.tensor(train_dataset), torch.tensor(test_dataset)), 0)
            
            print(coarse_labels.shape, dataset.shape)
            
            cl = torch.randperm(len(coarse_labels))
            train_size = int((train_size / 20)  * len(coarse_labels))
            test_size = int((test_size / 20)  * len(coarse_labels))
            val_size = int((val_size / 20)  * len(coarse_labels))
            
            print(train_size, test_size, val_size)
            
            train_cl = cl[:train_size]
            test_cl = cl[train_size: train_size+test_size]
            val_cl = cl[train_size+test_size:]
            
            train_data = dataset[train_cl].view(-1, 32, 32, 3)
            test_data = dataset[test_cl].view(-1, 32, 32, 3)
            val_data = dataset[val_cl].view(-1, 32, 32, 3)
            
            train_label = coarse_labels[train_cl]
            test_label = coarse_labels[test_cl]
            val_label = coarse_labels[val_cl]
            
            
            return train_data.permute(0, 3, 1, 2)/255, train_label, test_data.permute(0, 3, 1, 2)/255, test_label, val_data, val_label
        
            
            """
            
            print(train_data.shape, test_data.shape, val_data.shape)
            print(train_labels.shape, test_labels.shape, val_labels.shape)
            
            train_ds = DatasetCIFAR(train_data, train_labels)
            test_ds = DatasetCIFAR(test_data, test_labels)
            val_ds = DatasetCIFAR(val_data, val_labels)
            
            train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
            test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=True)
            val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=True)
            
            return train_dl, test_dl, val_dl
        
            """
        
        
    else:
        print("Type de dataset inconnu : ", dataset)
        return None
    
            



#train_data, train_label, test_data, test_label, val_data, val_label = get_CIFARFS()
#train_data, train_label, test_data, test_label, val_data, val_label = get_CIFARFS(14, 4, 2, "fc100")

