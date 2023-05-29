from torch.utils.data import DataLoader
from torchtext.data.functional import to_map_style_dataset
from .config import Config
from model import load_model
import json
import pandas as pd
import spacy
import torch
import tweepy

nlp = spacy.load('en_core_web_lg')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = Config()

client = tweepy.Client(
    bearer_token=config.BEARER_TOKEN,
    consumer_key=config.CONSUMER_KEY,
    consumer_secret=config.CONSUMER_SECRET,
    access_token=config.ACCESS_TOKEN,
    access_token_secret=config.ACCESS_TOKEN_SECRET,
)

with open('../../accounts.json') as f:
    accounts = json.load(f)


def process_tweets():
    tweets = get_tweets(n=10)
    tokenized_tweets = [nlp(tweet).vector for tweet in tweets]
    df = pd.DataFrame({'text': tweets, 'vector': tokenized_tweets})

    classify_dataset = to_map_style_dataset(df['vector'])
    test_dataloader = DataLoader(classify_dataset, batch_size=64, shuffle=True, collate_fn=collate_batch)

    predictions = predict(test_dataloader)
    df['label'] = predictions

    return df


def get_tweets(n):
    tweets = []
    for account in accounts:
        response = client.get_users_tweets(account['id'], max_results=n)
        tweets.extend([tweet.text for tweet in response.data])

    return tweets


model = load_model('class_model.pt')


def predict(dataloader):
    model.eval()
    preds = []

    with torch.no_grad():
        for idx, vector in enumerate(dataloader):
            predicted_label = model(vector)
            preds.extend(predicted_label.argmax(1).cpu().detach().numpy())

    return preds


def collate_batch(batch):
    embedding_list = [torch.tensor(_embedding, dtype=torch.float32) for _embedding in batch]
    embedding_list = torch.stack(embedding_list)

    return embedding_list.to(device)


with open('../labels.json') as f:
    labels = json.load(f)


def filter_tweets(topic):
    df = process_tweets()
    filtered_df = df.loc[df['label'] == labels.index(topic)]

    return filtered_df['text'].values


def get_top_categories(n):
    df = process_tweets()
    df['label'] = df['label'].apply(lambda x: labels[x])
    df = df.groupby('label')['label'].count().reset_index(name='count').sort_values(['count'], ascending=False)
    df = df.head(int(n))

    categories = df['label'].values.tolist()
    counts = df['count'].values.tolist()

    return categories, counts
