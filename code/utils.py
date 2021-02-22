# -*- coding: utf-8 -*-

import torch
#import cifar_dataset as cf
import numpy as np

def unpickle(file): 
    import pickle 
    with open(file, 'rb') as fo: 
        dict = pickle.load(fo, encoding='bytes') 
    return dict

def getSupportQuery(data, label, K=3, N=5):
    cl = np.random.choice(np.unique(label), K, replace=False)
    print("cl ! ", cl)
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





def sim_acc(d,classes,V):
    """ d : tensor de similarité (Nc * Nq,Nc)
        classes : tensor des classes (Nc * Nq)
        V : numpy numéro des classes (Nc)
    """
    
    #recup les prédictions
    indices = torch.argmax(d,dim = 1)
    indices_array = indices.numpy()
    y_chap = V[indices_array]
    
    #calcul accuracy
    acc = np.where(classes.numpy() == y_chap,1,0)
    
    return torch.true_divide(torch.sum(torch.tensor(acc)),d.shape[0]) * 100





def sample_query_support(data,labels,Ns,Nq,k):
    """ data : donnees
        labels : classes
        Ns : nombre d'exemples dans le support
        Ns : nombre d'exemples dans la query
        k : numéro de la classe 
    """
    
    print("size de data : ",data.shape)
    print("classe traité : ",k)
    print("labels : ",labels)
    #indices des exemples de la classe k  
    all_k_ind = torch.where(labels == k)[0]
    print("all k ind : ",all_k_ind)
    
    #choisir des exemples aléatoires pour le support
    supp_indices = np.random.choice(all_k_ind,Ns,replace=False)
    supp_exemples = data[supp_indices]
    supp_labels = labels[supp_indices]
    
    
    print("supp_ind : ",len(supp_indices))
    print("supp_exemples : ",supp_exemples.shape)
    print("sup : ",supp_labels)
    print()
    
    #enlever indices déjà choisis pour support 
    all_k_ind_q = torch.tensor(list(set(all_k_ind.tolist()) - set(supp_indices.tolist())))
    #print("all_ind_q shape : ",all_k_ind_q.shape)
    
    #choisir exemples aléatoires pour la query
    query_indices = np.random.choice(all_k_ind_q,Nq,replace=False)
    query_exemples = data[query_indices]
    query_labels = labels[query_indices]
    
    print("query_ind : ",len(query_indices))
    print("query_exemples : ",query_exemples.shape)
    print("query : ",query_labels)
    print()
    
    return supp_exemples,supp_indices,query_exemples,query_labels
   
    
def training_prototypical(data,labels,Nc,Ns,Nq,model,flag):
    """ data : données de train
        labels : classes
        Nc : nombre de classe par épisode
        Ns : nombre d'exemple dans le support de chaque classe
        Nq : nombre d'exemple dans le query de chaque classe
        flag : 0 si conv4, 1 si resnet12
    """
    
    #choisir les classes pour cet épisode
    
    V = np.random.choice(torch.unique(labels),Nc,replace = False)
    
    cos = torch.nn.CosineSimilarity(dim = 1)
    
    prototypes = []
    queries = []
    classes = []
    for k in V:
        supp_x,supp_y,query_x,query_y = sample_query_support(data,labels,Ns,Nq,k)
        
        #modèle sur le support 
        output = model.forward(supp_x,flag)
        print("output shape : ",output.shape)
        #print("output : ",output)
        vectors = torch.flatten(output,1,3)
        print("vectors shape : ",vectors.shape)
    
        ck = torch.sum(vectors,0) / Nc
        print("ck shape : ",ck.shape)
        
        prototypes.append(ck)
        queries.append(query_x)
        classes.append(query_y)
        
    tens_ck = torch.stack(prototypes)
    print("tens ck shape : ",tens_ck.shape)
    
    tens_queries = torch.cat(queries)
    print("tens_queries shape : ",tens_queries.shape)
    
    tens_classes = torch.cat(classes,dim = 0)
    print("tens_classes shape : ",tens_classes.shape)


    
    #modèle sur le query
    loss = 0
    total_sim = 0
    liste_d = []
    for i in range(Nc):
        
        
        #similarité entre les query et un prototype
        output_query = model.forward(tens_queries,flag)
        query_vectors = torch.flatten(output_query,1,3)
        
        print("\nquery_vectors shape : ",query_vectors.shape)
        
        ck_extend = torch.stack([tens_ck[i]] * (Nc * Nq))
        print("ck_extend shape : ",ck_extend.shape)
        
        sim = cos(query_vectors,ck_extend)
        print("ck : ",ck)
        print("dim de sim : ",sim.shape)
        
        liste_d.append(sim)
        
    
    #somme des similarités totales
    tens_sim = torch.stack(liste_d,dim = 1)
    print("\ntens_sim shape : ",tens_sim.shape)
    
    total_sim = torch.log(torch.sum(-torch.exp(tens_sim)))
    
    print("labels shape : ",tens_classes.shape)
    print("labels : ",tens_classes)
    
    #similarité entre une query et le prototype associé
    j = 0
    m = 0
    for i in range(len(tens_sim)):
        if i >= m + Nq:
            j += 1
            m += i
        loss += tens_sim[i,j]
        
    acc = sim_acc(tens_sim,tens_classes,V)
    
    return (loss + total_sim)/(Nc * Nq),acc


def training(train_data, train_labels, test_data, test_labels, nb_series):

  
  # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  # train_data = train_data.to(device)
  # train_labels = train_labels.to(device)
  # test_data = test_data.to(device)
  # test_labels = test_labels.to(device)


  model = ConstellationNet(nb_clusters=8, nb_heads=1, nb_epochs=1, beta=1, lbda=1, n_channels_start=3, n_channels_convo=64, output_size=100)
  #model = model.float()
  #model = model.to(device)
  criterion = torch.nn.CrossEntropyLoss()
  optim = torch.optim.SGD(model.parameters(), lr=1, momentum=0.9, weight_decay=0.0005)

  loss_train = []
  acc_train = []

  loss_test = []
  acc_test = []

 
  optim.zero_grad()

  """
    Partie training avec toute la partie Train
  """
  features = model(train_data)
  l = criterion(features, train_labels)
  l.backward()
  loss_train.append(l.item())

  optim.step()

  """
    Partie test avec ProtoNet-Based Framework
  """
  print("Debut partie test avec évaluation en séries")

  all_acc = []

  for i in tqdm(range(nb_series)):
    
    supp_data, supp_labels, query_data, query_labels = getSupportQuery(test_data, test_labels)
    print(supp_data.shape, supp_labels.shape, query_data.shape, query_labels.shape)
    acc_mean = 0

    for _ in range(10):

      """
        Training support-set
      """
      optim.zero_grad()
      features = model(supp_data)
      #print("f s :", features.shape)
      l = criterion(features, supp_labels)
      l.backward()
      optim.step()

      """
        Testing query-set
      """
      x_query = model(query_data)
      #print("xquery : ", x_query)
      pred = getSimilarity(features, supp_labels, x_query)
      acc = pred.eq(query_labels).float().mean()
      print("Acc : ", acc)
      all_acc.append(acc)
    


  return all_acc
        
        





    
        


    
    
 

    
        
        
        
        