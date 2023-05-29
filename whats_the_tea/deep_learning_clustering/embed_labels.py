from sentence_transformers import SentenceTransformer
import json
import torch

with open('labels.json') as f:
    labels = json.load(f)

labels = [label.lower() for label in labels]
model = SentenceTransformer('all-MiniLM-L6-v2')
choice_embeddings = torch.stack([model.encode(x, convert_to_tensor=True) for x in labels])
torch.save(choice_embeddings, 'choices.pt')
