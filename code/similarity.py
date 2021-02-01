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
        all_c.append(torch.mean(features_l, 0))
        
    cos_sim = cosine_similarity(torch.cat(all_c, 0), x_query)
    cos_sim /= torch.sum(torch.exp(cos_sim))
    
    pred = all_labels[torch.argmax(cos_sim)]
    
    return pred
    