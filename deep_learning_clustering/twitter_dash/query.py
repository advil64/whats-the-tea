from collections import Counter
from model import *  # load_model
from .config import Config
import json
import spacy
import numpy as np
import torch
import tweepy

nlp = spacy.load('en_core_web_lg')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config = Config()
batch_size = 64

client = tweepy.Client(
    bearer_token=config.BEARER_TOKEN,
    consumer_key=config.CONSUMER_KEY,
    consumer_secret=config.CONSUMER_SECRET,
    access_token=config.ACCESS_TOKEN,
    access_token_secret=config.ACCESS_TOKEN_SECRET,
)

with open('../../accounts.json') as f:
    accounts = json.load(f)

with open('../labels.json') as f:
    labels = json.load(f)

model = TweetClassifier().to(device)  # load_model('class_model.pt')


def get_tweets(n):
    tweets = []
    for account in accounts:
        response = client.get_users_tweets(account['id'], max_results=n)
        tweets.extend([tweet.text for tweet in response.data])

    return tweets


def filter_tweets(topic):
    tweets = get_tweets(10)
    tokenized_tweets = np.array([nlp(tweet).vector for tweet in tweets])
    tweet_tensors = torch.tensor(tokenized_tweets).unsqueeze(1)

    predictions = model.predict(tweet_tensors, batch_size=batch_size)
    indices = [i for i, value in enumerate(predictions) if value == labels.index(topic)] if labels.index(
        topic) in predictions else None

    return [tweets[i] for i in indices] if indices else None


def get_top_categories(n):
    tweets = get_tweets(n)
    tokenized_tweets = np.array([nlp(tweet).vector for tweet in tweets])
    tweet_tensors = torch.tensor(tokenized_tweets).unsqueeze(1)

    predictions = model.predict(tweet_tensors, batch_size=batch_size)
    predictions = [labels[prediction] for prediction in predictions]

    counts = Counter(predictions)
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)

    return zip(*sorted_counts)
