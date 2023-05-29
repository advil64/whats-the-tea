from torch.utils.data import DataLoader
from torchtext.data.functional import to_map_style_dataset
from .config import Config
from ..model import load_model
import json
import pandas as pd
import spacy
import torch
import tweepy

nlp = spacy.load('en_core_web_lg')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

config = Config()

client = tweepy.Client(
    bearer_token=config.BEARER_TOKEN,
    consumer_key=config.CONSUMER_KEY,
    consumer_secret=config.CONSUMER_SECRET,
    access_token=config.ACCESS_TOKEN,
    access_token_secret=config.ACCESS_TOKEN_SECRET,
)

with open('accounts.json') as f:
    accounts = json.load(f)


def process_tweets():
    tweets = get_tweets(n=10, batch_size=64)
    embeddings = [preprocess(tweet) for tweet in tweets]
    df = pd.DataFrame({'text': tweets, 'vector': embeddings})

    classify_dataset = to_map_style_dataset(df['vector'])
    test_dataloader = DataLoader(classify_dataset, batch_size=64, shuffle=True, collate_fn=collate_batch)

    predictions = predict(test_dataloader)
    df['label'] = predictions

    return df


def get_tweets(n, batch_size):
    tweets = []
    for account in accounts:
        response = client.get_users_tweets(account['id'], max_results=n)
        tweets.extend([tweet.text for tweet in response.data])

    return tweets[: (len(tweets) // batch_size) * batch_size]


def preprocess(text):
    return nlp(text).vector


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


label_mapping = {
    'ARTS': 0,
    'ARTS & CULTURE': 1,
    'BLACK VOICES': 2,
    'BUSINESS': 3,
    'COLLEGE': 4,
    'COMEDY': 5,
    'CRIME': 6,
    'CULTURE & ARTS': 7,
    'DIVORCE': 8,
    'EDUCATION': 9,
    'ENTERTAINMENT': 10,
    'ENVIRONMENT': 11,
    'FIFTY': 12,
    'FOOD & DRINK': 13,
    'GOOD NEWS': 14,
    'GREEN': 15,
    'HEALTHY LIVING': 16,
    'HOME & LIVING': 17,
    'IMPACT': 18,
    'LATINO VOICES': 19,
    'MEDIA': 20,
    'MONEY': 21,
    'PARENTING': 22,
    'PARENTS': 23,
    'POLITICS': 24,
    'QUEER VOICES': 25,
    'RELIGION': 26,
    'SCIENCE': 27,
    'SPORTS': 28,
    'STYLE': 29,
    'STYLE & BEAUTY': 30,
    'TASTE': 31,
    'TECH': 32,
    'THE WORLDPOST': 33,
    'TRAVEL': 34,
    'U.S. NEWS': 35,
    'WEDDINGS': 36,
    'WEIRD NEWS': 37,
    'WELLNESS': 38,
    'WOMEN': 39,
    'WORLD NEWS': 40,
    'WORLDPOST': 41
}


def filter_tweets(topic):
    df = process_tweets()
    filtered_df = df.loc[df['label'] == label_mapping[topic]]

    return filtered_df['text'].values


def get_top_categories(n):
    inverted_label_mapping = {v: k for k, v in label_mapping.items()}

    df = process_tweets()
    df['label'] = df['label'].apply(lambda x: inverted_label_mapping[x])
    df = df.groupby('label')['label'].count().reset_index(name='count').sort_values(['count'], ascending=False)
    df = df.head(int(n))

    categories = df['label'].values.tolist()
    counts = df['count'].values.tolist()

    return categories, counts
