import argparse
import json
import torch
import pathlib
from sentence_transformers import SentenceTransformer, util
from time import time
from tqdm import tqdm

PREFIX = pathlib.Path('/common/users/shared/cs543_fall22_group3/combined/combined_raw')


def load_labels(file_path):
    with open(file_path) as f:
        labels = json.load(f)

    return [label.lower() for label in labels]


def load_articles(file_path):
    with open(file_path) as f:
        data = f.readlines()

    return [json.loads(sample)['selected_text'] for sample in tqdm(data)]


def save_results(file_path, articles, assigned_labels):
    with open(file_path, 'w') as f:
        json.dump([{'article': article, 'label': label} for article, label in zip(articles, assigned_labels)], f)


def main(args):
    inpath = PREFIX / args.infile
    outfile = args.outpath / args.infile

    t0 = time()

    labels = load_labels('labels.json')
    print(f'Loaded {len(labels)} labels.')

    articles = load_articles(inpath)
    print(f'Loaded {len(articles)} articles.')

    model = SentenceTransformer('all-MiniLM-L6-v2')
    label_embeddings = torch.stack([model.encode(x, convert_to_tensor=True) for x in labels])
    article_embeddings = model.encode(articles, convert_to_tensor=True)

    cos_sims = util.cos_sim(article_embeddings, label_embeddings)
    max_idxs = torch.argmax(cos_sims, dim=1)
    assigned_labels = [labels[x] for x in max_idxs]

    save_results(outfile, articles, assigned_labels)
    print(f'Results saved to {outfile}.')

    print(f'Done. Took {round(time() - t0, 5)} s.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--infile', type=pathlib.Path, required=True)
    parser.add_argument('--outpath', type=pathlib.Path, required=True)
    args = parser.parse_args()

    main(args)
