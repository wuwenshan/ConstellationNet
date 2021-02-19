# -*- coding: utf-8 -*-
import torch
from scipy.spatial import distance_matrix
import math
from sklearn.metrics.pairwise import cosine_similarity

class ConstellationNet(torch.nn.Module):
    
    def __init__(self, k, nb_head, n_channels_start, n_channels_convo, n_channels_concat, n_channels_one_one):
        super(ConstellationNet, self).__init__()
        
        self.nb_cluster = k
        self.nb_head = nb_head
        self.n_channels_start = n_channels_start
        self.n_channels_convo = n_channels_convo
        self.n_channels_concat = n_channels_concat
        self.n_channels_one_one = n_channels_one_one
        
        
        #param de conv4
        # channels en entrée (RGB), 16 filtres, filtre 3* 3
        self.conv_trois_trois = torch.nn.Conv2d(n_channels_start,n_channels_convo,3,padding = 1)
        
        #conv 1x1 finale
        self.conv_un_un = torch.nn.Conv2d(n_channels_concat,n_channels_one_one,1)
        
        self.relu = torch.nn.ReLU()
        self.max_pool = torch.nn.MaxPool2d(2)
        
        
        #param du resnet
        """
        #conv 1 x 1
        self.conv_un_un_res1 = torch.nn.Conv2d(k + 64,n_channels_one_one,1)
        self.conv_un_un_res2 = torch.nn.Conv2d(k + 128,n_channels_one_one,1)
        self.conv_un_un_res3 = torch.nn.Conv2d(k + 256,n_channels_one_one,1)
        self.conv_un_un_res4 = torch.nn.Conv2d(k + 512,n_channels_one_one,1)
        
        #residual bloc 1
        self.res1conv1 = torch.nn.Conv2d(n_channels_start,64,3,padding = 1)
        #self.res1conv23 = torch.nn.Conv2d(64,64,3)
        self.batch_norm_res1 = torch.nn.BatchNorm2d(64)
        
        #residual block 2
        self.res2conv1 = torch.nn.Conv2d(n_channels_one_one,128,3)
        #self.res2conv23 = torch.nn.Conv2d(128,128,3)
        self.batch_norm_res2 = torch.nn.BatchNorm2d(128)
        
        #residual block 3
        self.res3conv1 = torch.nn.Conv2d(n_channels_one_one,256,3)
        #self.res3conv23 = torch.nn.Conv2d(128,256,3)
        self.batch_norm_res3 = torch.nn.BatchNorm2d(256)
        
        #residual block 4
        self.res4conv1 = torch.nn.Conv2d(n_channels_one_one,512,3)
        #self.res4conv23 = torch.nn.Conv2d(512,512,3)
        self.batch_norm_res4 = torch.nn.BatchNorm2d(512)
        """
        #on commence avec 64 (2^6) channels et on double pour les blocks suivants
        
    
        
    def forward(self, x):
        for i in range(4):
            print(f"X initial (step {i}): ", x.shape)

            # convolutionnal feature map
            cfm = self.conv(x)

            print("After conv : ", cfm.shape)
            
            # cell relation modeling
            constell_module = self.constell(cfm)

            print("After constell : ", constell_module.shape)
            
            x = self.concatenation(cfm, constell_module, self.conv_un_un)
            
            print("Post concat : ", x.shape)

            x = x.float()
    
        return x # self.similarity(x)
  
    
    
    
    
    
    
    def conv(self,X):
        """ bloc simple de convolution pour Conv-4 """
        
        #convolution 3 * 3
        convo = self.conv_trois_trois(X)
        #print("Après la convo shape de X : ",convo.shape)
        
        
        #batch_norm
        batch_norm_convo = torch.nn.BatchNorm2d(convo.shape[1])(convo)
        #print("Après normalisation : ",batch_norm_convo.shape)
        
        
        #relu
        relu_convo = self.relu(batch_norm_convo)
        #print("Après relu : ",relu_convo.shape)
        
        
        #max pool
        convo_feature_map = self.max_pool(relu_convo)
        #print("Après max pool  : ",convo_feature_map.shape)
        
        
        return convo_feature_map
        
    
    def conv4_constell(self,X):
        """ Conv-4 backbone avec module Constell """  
        for i in range(4):
            
            #convolution
            features_map = self.conv(X)
            
            #constellation
            X_constell = torch.randn(X.shape[0],self.nb_cluster,features_map.shape[2],features_map.shape[3])
            #X_constell = self.constell(features_map)
            
            #concatenation
            X_concat = self.concatenation(X_constell,features_map,self.conv_un_un)
            
            X = X_concat
        print("X_concat size : ",X_concat.shape)
            
        return X

    def concatenation(self, X_feature_map, X_constell, conv1):
        """ partie concaténation entre Features Map et la sortie du Constellation Module """
        
        """
        print("Dans concat")
        print("X constell shape : ",X_constell.shape)
        print("X feature map shape :",X_feature_map.shape)
        """
        
        concat = torch.cat((X_constell,X_feature_map),1)
        
        """
        print("concat des deux : ",concat.shape)
        print()
        """
        
        #conv 1x1
        conv_one_one = conv1(concat)
        
        #BatchNorm
        concat_batch_norm = torch.nn.BatchNorm2d(conv_one_one.shape[1])(conv_one_one)
        
        #Relu
        relu_concat = self.relu(concat_batch_norm)
        
        return relu_concat
        
    
    
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
        X_constell = torch.randn(X.shape[0],self.nb_cluster,X.shape[2],X.shape[3])
        #print("X_constell shape : ",X_constell.shape)
        
        #concat Feature map et Constellation
        
        X_concat = self.concatenation(X_constell,X_feature_map,conv1)
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
            X_i_constell = torch.randn(X_convo_i.shape[0],self.nb_cluster,X_convo_i.shape[2],X_convo_i.shape[3])
            X_concat = self.concatenation(X_i_constell,X_convo_i_feature_map,conv1)
            #print("resbloc, shape de x concat : ",X_concat.shape)
        
        
        
        return X_concat
    
    
    
    def init_convo_layers(self):
        """ initialiser les layers de convolutions et de normalisation pour les différents blocs résiduels """
        
        #convo 3 x 3  avec n_channels = 64,128,256,512
        convo_trois_trois_layers = []
        convo_one_one_layers = []
        batch_norm_layers = []
        
        i = 1
        n_channels_res_convo = 64
        
        convo_trois_trois_layers.append(torch.nn.Conv2d(self.n_channels_start,n_channels_res_convo,3,padding = 1))
        convo_one_one_layers.append(torch.nn.Conv2d(self.nb_cluster + n_channels_res_convo,self.n_channels_one_one,1))
        batch_norm_layers.append(torch.nn.BatchNorm2d(n_channels_res_convo))
        
        while i < 4:
            
            #nombre de filtres doublés d'un bloc à l'autre
            n_channels_res_convo *= 2
            convo_trois_trois_layers.append(torch.nn.Conv2d(self.n_channels_start,n_channels_res_convo,3,padding = 1))
            convo_one_one_layers.append(torch.nn.Conv2d(self.nb_cluster + n_channels_res_convo,self.n_channels_one_one,1))
            batch_norm_layers.append(torch.nn.BatchNorm2d(n_channels_res_convo))
            
            i += 1
            
        
        
        self.convo_trois_trois_layers = convo_trois_trois_layers
        self.convo_one_one_layers = convo_one_one_layers
        self.batch_norm_layers = batch_norm_layers
        
        #return convo_trois_trois_layers,convo_one_one_layers
    
    
    
    
    
    
    def resnet12_constell(self,X):
        """ ResNet-12 backbone avec module Constell """
        self.init_convo_layers()
        
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
        
    
   
    
    
    
    
        
    
    def constell(self, cfm):
        # distance map par soft k-means
        d_map = self.cellFeatureClustering(cfm, k=self.nb_cluster, epoch=10)

        print("distance map : ", d_map.shape)
        
        # positional encoding
        pos_enc = self.positionalEncoding(b=1, h=16, w=16, c=self.nb_cluster)

        print("Positional encoding : ", pos_enc.shape)
        
        # output feature fa avec un self attention mechanism
        fa = self.multiHeadAtt(d_map, pos_enc, self.nb_head)

        print("Multi Head Att : ", fa.shape)
        h = int(math.sqrt(fa.shape[1]))
        fa = fa.view(fa.shape[0], h, h, fa.shape[2]).permute(0, 3, 1, 2)

        print("Multi Head Att : ", fa.shape)
    
        return fa
    
    
    """
        Cette étape de clustering se compose d'un soft k-means produisant une "distance map"
        Indication du papier :
            - beta > 0
            - lambda = 1.0
    """
    
    
    def cellFeatureClustering(self, cellFeature, k, epoch, beta=100, lbda=1.0):
    
        batch_size, nb_chan, h_size, w_size = cellFeature.shape
        
        # initialisation
        s = torch.zeros(k)
        v = torch.randn(k, nb_chan)
        cellFeature = cellFeature.permute(1, 0, 2, 3).contiguous().view(nb_chan, -1)

        
        for _ in range(epoch):

            # cluster assignment
            d = torch.tensor(distance_matrix(cellFeature.T.detach().numpy(), v.detach().numpy()))

            m = torch.zeros(d.shape)
            for i in range(len(m)):
                m[i] = torch.exp(-beta * d[i]) / torch.sum( torch.exp(-beta * d[i]) )
                
            v_p = torch.zeros(v.shape)
            for i in range(len(v_p)):
                v_p[i] = torch.sum(m[:,i]) * torch.sum(cellFeature[i]) / torch.sum(m[:,i])
            
            # centroid movement
            delta_s = torch.sum(m, 0)
            mu = lbda / (s + delta_s)
            v = (1 - mu.unsqueeze(1)) * v + mu.unsqueeze(1) * v_p
            
            # counter update
            s += delta_s
        
        d_map = d.view(batch_size, h_size, h_size, d.shape[1])
        d_map = d_map.permute(0, 3, 1, 2)
        
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
        
        wq = torch.nn.Linear(k, k)
        wq = wq.double()

        wk = torch.nn.Linear(k, k)
        wk = wk.double()

        wv = torch.nn.Linear(k, k)
        wv = wv.double()

        fq = wq(f1)
        fk = wk(f1)
        fv = wv(f1_p)

        sm = torch.nn.Softmax(dim=2)

        fa = torch.max( sm((torch.bmm(fq, fk.permute(0, 2, 1)) / math.sqrt(k) )), 2)[0].unsqueeze(2) * fv
        
        return fa
    
    def multiHeadAtt(self, d_map, p_enc, head):
        k = d_map.shape[1]
        w = torch.nn.Linear(k*head, k)
        w = w.double()
        
        all_head = []
        for _ in range(head):
            all_head.append(self.oneHeadAtt(d_map, p_enc))
            
        return w(torch.cat(all_head, 2))
    
    """
    def similarity(self, featuresMap):
        fm = torch.nn.AvgPool2d(featureMap, kernel_size=2, stride=2)
        emb = fm.view(len(fm), -1)
        query = emb[-1]
        support = emb[:-1]
        cos_sim = cosine_similarity(support, query)
        pred = torch.argmax(torch.tensor(cos_sim))
        
        return pred
    """