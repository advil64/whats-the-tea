from pathlib import Path
from sentence_transformers import SentenceTransformer, util
from time import time
from tqdm import tqdm
import argparse
import json
import os
import torch


def load_topics(file_path):
    with open(file_path) as f:
        topics = json.load(f)

    return topics


def load_articles(file_path):
    file_list = os.listdir(file_path)

    articles = []
    for filename in tqdm(file_list):
        if filename.endswith('.json'):
            with open(os.path.join(file_path, filename)) as f:
                for line in f:
                    json_object = json.loads(line)
                    articles.append(json_object['article'])

    return articles


def save_results(file_path, articles, assigned_topics):
    with open(file_path, 'w') as f:
        json.dump([{'article': article, 'topic': topic} for article, topic in zip(articles, assigned_topics)], f)


def main(args):
    inpath = args.inpath
    outpath = args.outpath

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    t0 = time()

    topics = load_topics('topics.json')
    print(f'Loaded {len(topics)} topics.')

    articles = load_articles(inpath)
    print(f'Loaded {len(articles)} articles.')

    model = SentenceTransformer('all-MiniLM-L6-v2')
    topic_embeddings = model.encode(topics, show_progress_bar=True, convert_to_tensor=True, device=device)
    article_embeddings = model.encode(articles, show_progress_bar=True, convert_to_tensor=True, device=device)

    cos_sims = util.cos_sim(article_embeddings, topic_embeddings)
    max_idxs = torch.argmax(cos_sims, dim=1)
    assigned_topics = [topics[x] for x in max_idxs]

    save_results(outpath, articles, assigned_topics)
    print(f'Results saved to {outpath}.')

    print(f'Done. Took {round(time() - t0, 5)} s.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inpath', type=Path, required=True)
    parser.add_argument('--outpath', type=Path, required=True)
    args = parser.parse_args()

    main(args)
