import torch
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer, util

labels = ['arts', 'arts & culture', 'black voices', 'business', 'college', 
        'comedy', 'crime', 'culture & arts', 'divorce', 'education', 'entertainment', 
        'environment', 'fifty', 'food & drink', 'good news', 'green', 
        'healthy living', 'home & living', 'impact', 'latino voices', 
        'media', 'money', 'parenting', 'parents', 'politics', 'queer voices', 
        'religion', 'science', 'sports', 'style', 'style & beauty', 'taste', 
        'tech', 'the worldpost', 'travel', 'u.s. news', 'weddings', 'weird news', 
        'wellness', 'women', 'world news', 'worldpost']

model = SentenceTransformer('all-MiniLM-L6-v2')

choice_embeddings = torch.stack([model.encode(x, convert_to_tensor=True) for x in labels])

torch.save(choice_embeddings, 'choices.pt')

# with open('choices.pkl', 'wb') as f:
#     pickle.dump(choice_embeddings)
