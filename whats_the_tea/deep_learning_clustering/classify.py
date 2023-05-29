import argparse
import json
import torch
import pathlib
from sentence_transformers import SentenceTransformer, util
from time import time
from tqdm import tqdm

with open('labels.json') as f:
    labels = json.load(f)

labels = [label.lower() for label in labels]

PREFIX = pathlib.Path('/common/users/shared/cs543_fall22_group3/combined/combined_raw')

parser = argparse.ArgumentParser()
parser.add_argument('--infile', type=pathlib.Path, required=True)
parser.add_argument('--outpath', type=pathlib.Path, required=True)
args = parser.parse_args()

inpath = PREFIX / args.infile
outfile = args.outpath / args.infile

t0 = time()

with open(inpath, 'r') as f:
    data = f.readlines()

print(f'Size of data = {len(data)}.')

model = SentenceTransformer('all-MiniLM-L6-v2')
choices = torch.load('choices.pt')

queries = [json.loads(sample)['selected_text'] for sample in tqdm(data)]

query_embedding = model.encode(queries, convert_to_tensor=True)

cos_sims = util.cos_sim(query_embedding, choices)
max_idxs = torch.argmax(cos_sims, dim=1)

assigned_labels = [labels[x] for x in max_idxs]

with open(outfile, 'w') as f:
    json.dump([{'query': q, 'label': l} for q, l in zip(queries, assigned_labels)], f)

print(f'Done. Took {round(time() - t0, 5)} s.')
