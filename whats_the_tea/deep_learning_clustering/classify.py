import json
import torch
import pathlib
import argparse
import numpy as np
from time import time
from tqdm import tqdm
# from transformers import T5Tokenizer, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer, util

LABELS = ['arts', 'arts & culture', 'black voices', 'business', 'college', 
        'comedy', 'crime', 'culture & arts', 'divorce', 'education', 'entertainment', 
        'environment', 'fifty', 'food & drink', 'good news', 'green', 
        'healthy living', 'home & living', 'impact', 'latino voices', 
        'media', 'money', 'parenting', 'parents', 'politics', 'queer voices', 
        'religion', 'science', 'sports', 'style', 'style & beauty', 'taste', 
        'tech', 'the worldpost', 'travel', 'u.s. news', 'weddings', 'weird news', 
        'wellness', 'women', 'world news', 'worldpost']

# device = torch.device('cuda')

# "part-00860-1d603278-60b7-4e2d-980f-bc3a1287f222-c000.json"

PREFIX = pathlib.Path("/common/users/shared/cs543_fall22_group3/combined/combined_raw")

parser = argparse.ArgumentParser()
parser.add_argument('--infile', type=pathlib.Path, required=True)
parser.add_argument('--outpath', type=pathlib.Path, required=True)
args = parser.parse_args()

inpath = PREFIX / args.infile
outfile = args.outpath / args.infile

t0 = time()

with open(inpath, 'r') as f:
    data = f.readlines()

print(f"Size of data = {len(data)}.")

model = SentenceTransformer('all-MiniLM-L6-v2')
choices = torch.load('choices.pt')

queries = [json.loads(sample)['selected_text'] for sample in tqdm(data)]

query_embedding = model.encode(queries, convert_to_tensor=True)
# print(query_embedding.shape)

cos_sims = util.cos_sim(query_embedding, choices)
max_idxs = torch.argmax(cos_sims, dim=1)

# print(max_idxs)
assigned_labels = [LABELS[x] for x in max_idxs]

with open(outfile, 'w') as f:
    json.dump([{'query': q, 'label': l} for q, l in zip(queries, assigned_labels)], f)

print(f"Done. Took {round(time() - t0, 5)} s.")