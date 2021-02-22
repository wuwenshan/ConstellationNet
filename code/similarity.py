# -*- coding: utf-8 -*-

import torch
from sklearn.metrics.pairwise import cosine_similarity

# prédictions basées sur le ProtoNet-Based Framework

# features apprises lors de l'apprentissage
# labels correspondant à chaque image
# x_query : image sous la forme de feature 
def getSimilarity(features, labels, x_query):
    
    all_c = [] # cf prototype c_k
    all_labels = torch.unique(labels)
    
    for l in all_labels:
        features_l = features[torch.where(labels == l)]
        #print("mean : ", torch.mean(features_l, 0))
        all_c.append(torch.mean(features_l, 0).unsqueeze(0))

    cos_sim = torch.tensor(cosine_similarity(torch.cat(all_c, 0).detach().numpy(), x_query.detach().numpy())).T
    p = torch.exp(cos_sim) / torch.sum(torch.exp(cos_sim))
    
    pred = all_labels[torch.argmax(p, 1)]
    
    return pred
    