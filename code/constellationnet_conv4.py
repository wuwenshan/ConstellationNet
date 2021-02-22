# -*- coding: utf-8 -*-
import torch
from scipy.spatial import distance_matrix
import math
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import numpy as np

class ConstellationNet_conv4(torch.nn.Module):
    
    def __init__(self, nb_clusters, nb_heads, nb_epochs, beta, lbda, n_channels_start, n_channels_convo, output_size):
        super(ConstellationNet, self).__init__()
        
        self.nb_cluster = nb_clusters
        self.nb_head = nb_heads
        self.nb_epoch = nb_epochs
        self.beta = beta
        self.lbda = lbda
        

        self.concatPart = torch.nn.Sequential(
            torch.nn.Conv2d(n_channels_convo+nb_clusters, n_channels_start, 1),
            torch.nn.BatchNorm2d(n_channels_start),
            torch.nn.ReLU()
        )
        

        self.conv4 = torch.nn.Sequential(
          torch.nn.Conv2d(n_channels_start, n_channels_convo, 3, padding = 1),
          torch.nn.BatchNorm2d(n_channels_convo),
          torch.nn.ReLU(),
          torch.nn.MaxPool2d(2)
        )

        self.avg_pool = torch.nn.AvgPool2d(2)
        self.embeddings = torch.nn.Linear(3, output_size)

        self.wq = torch.nn.Linear(nb_clusters, nb_clusters)
        self.wk = torch.nn.Linear(nb_clusters, nb_clusters)
        self.wv = torch.nn.Linear(nb_clusters, nb_clusters)

        self.w = torch.nn.Linear(nb_clusters*nb_heads, nb_clusters)

    
        
    def forward(self, x):
        for i in tqdm(range(4)):
            #print(f"X initial (step {i}): ", x.shape)

            # convolutionnal feature map
            cfm = self.conv4(x)

            #print("After conv : ", cfm.shape)
            
            # cell relation modeling
            constell_module = self.constell(cfm)

            #print("After constell : ", constell_module.shape)
            x_concat = torch.cat((cfm, constell_module), 1)
            x = self.concatPart(x_concat)
            
            #print("Post concat : ", x.shape)

            x = x.float()
    
        x = self.avg_pool(x)
        x_emb = self.embeddings(x.view(len(x), -1))
  
        return x_emb # self.similarity(x)
  
  
    
    def constell(self, cfm):
        # distance map par soft k-means
        d_map = self.cellFeatureClustering(cfm)

        #print("distance map : ", d_map.shape)
        h = d_map.shape[2]
        b = d_map.shape[0]

        # positional encoding
        pos_enc = self.positionalEncoding(b, h, h, c=self.nb_cluster)

        #print("Positional encoding : ", pos_enc.shape)
        
        # output feature fa avec un self attention mechanism
        fa = self.multiHeadAtt(d_map, pos_enc)

        #print("Multi Head Att : ", fa.shape)
        h = int(math.sqrt(fa.shape[1]))
        fa = fa.view(fa.shape[0], h, h, fa.shape[2]).permute(0, 3, 1, 2)

        #print("Multi Head Att : ", fa.shape)
    
        return fa
    
    
    """
        Cette étape de clustering se compose d'un soft k-means produisant une "distance map"
        Indication du papier :
            - beta > 0
            - lambda = 1.0
    """
    
    
    def cellFeatureClustering(self, cellFeature):
        batch_size, nb_chan, h_size, w_size = cellFeature.shape
        
        # initialisation
        s = torch.zeros(self.nb_cluster)
        v = torch.randn(self.nb_cluster, nb_chan)
        cellFeature = cellFeature.permute(1, 0, 2, 3).contiguous().view(nb_chan, -1).T
        #v_ind = torch.randperm(len(cellFeature))[:k]
        #v = cellFeature[v_ind]

        #print("v : ", v)

        #print("cellFeature : ", cellFeature)

        for _ in range(self.nb_epoch):

            # cluster assignment
            #d = torch.pow( torch.tensor(distance_matrix(cellFeature.detach().numpy(), v.detach().numpy())), 2 )

            cdist = torch.cdist(cellFeature, v, 2, 'use_mm_for_euclid_dist')
            #print("cdist : ", cdist.shape)
            #print("cdist : ", cdist)
            #print("d : ", d)
            #print("m_i : ", torch.exp(-beta * d[0]), torch.sum( torch.exp(-beta * d[0]) ))

            m = torch.zeros(cdist.shape)
            for i in range(len(m)):
                m[i] = torch.exp(-self.beta * cdist[i]) / torch.sum( torch.exp(-self.beta * cdist[i]) )

            #print("m : ", m)
            m = torch.tensor( np.nan_to_num(m.detach().numpy(), nan=0.001, posinf=1, neginf=1) )
            
                
            v_p = torch.zeros(v.shape)
            for i in range(len(v_p)):
              #print(torch.sum(m[:,i]))
              v_p[i] = torch.sum(m[:,i]) * torch.sum(cellFeature[i]) / torch.sum(m[:,i])

            v_p = torch.tensor( np.nan_to_num(v_p.detach().numpy(), nan=0.001, posinf=1, neginf=1) )
            #print("v_p : ", v_p)
            
            # centroid movement
            delta_s = torch.sum(m, 0)
            #print("delta s ", delta_s)
            mu = self.lbda / (s + delta_s)
            mu = torch.tensor( np.nan_to_num(mu.detach().numpy(), nan=0.001, posinf=1, neginf=1) )
            #print("mu : ", mu)
            v = (1 - mu.unsqueeze(1)) * v + mu.unsqueeze(1) * v_p
            v = torch.tensor( np.nan_to_num(v.detach().numpy(), nan=0.001, posinf=1, neginf=1) )
            #print("v : ", v)
            # counter update
            s += delta_s
            s = torch.tensor( np.nan_to_num(s.detach().numpy(), nan=0.001, posinf=1, neginf=1) )
            #print("s : ", s)
        
        d_map = cdist.view(batch_size, h_size, h_size, cdist.shape[1])
        d_map = d_map.permute(0, 3, 1, 2)

        #print("d map : ", d_map)
        
        return d_map


    def positionalEncoding(self, b, h, w, c):
        orig_c = c
        c = int(torch.ceil(torch.tensor(c/2)))
        div_term = torch.exp(torch.arange(0., c, 2) * (-(math.log(10000.0) / c)))
        pos_h = torch.arange(0, h)
        pos_w = torch.arange(0, w)
        sin_inp_h = torch.einsum("i,j->ij", pos_h, div_term)
        sin_inp_w = torch.einsum("i,j->ij", pos_w, div_term)
        emb_h = torch.cat((sin_inp_h.sin(), sin_inp_h.cos()), dim=-1).unsqueeze(1)
        emb_w = torch.cat((sin_inp_w.sin(), sin_inp_w.cos()), dim=-1)
        emb = torch.zeros((b,h,w,c*2))
        emb[:,:,:,:c] = emb_h
        emb[:,:,:,c:2*c] = emb_w

        return emb[:,:,:,:orig_c].permute(0, 3, 1, 2)


    def oneHeadAtt(self, d_map, p_enc):
        b = d_map.shape[0]
        k = d_map.shape[1]
        f1 = (d_map + p_enc).view(b, k, -1).permute(0, 2, 1)
        f1_p = d_map.view(b, k, -1).permute(0, 2, 1)
        fq = self.wq(f1)
        fk = self.wk(f1)
        fv = self.wv(f1_p)

        sm = torch.nn.Softmax(dim=2)

        fa = torch.max( sm((torch.bmm(fq, fk.permute(0, 2, 1)) / math.sqrt(k) )), 2)[0].unsqueeze(2) * fv
        
        return fa
    
    def multiHeadAtt(self, d_map, p_enc):
        k = d_map.shape[1]
        
        all_head = []
        for _ in range(self.nb_head):
            all_head.append(self.oneHeadAtt(d_map, p_enc))
            
        return self.w(torch.cat(all_head, 2))
    
