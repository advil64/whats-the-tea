from collections import Counter
from model import TweetClassifier  # load_model
from .config import Config
import json
import numpy as np
import spacy
import torch
import tweepy

nlp = spacy.load('en_core_web_lg')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TweetClassifier().to(device)  # load_model('class_model.pt')
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

with open('../topics.json') as f:
    topics = json.load(f)


def get_tweets(n):
    tweets = []
    for account in accounts:
        response = client.get_users_tweets(account['id'], max_results=n)
        tweets.extend([tweet.text for tweet in response.data])

    return tweets


def process_tweets(tweets):
    tokenized_tweets = np.array([nlp(tweet).vector for tweet in tweets])
    tweet_tensors = torch.tensor(tokenized_tweets).unsqueeze(1)

    return tweet_tensors


def filter_tweets(topic):
    tweets = get_tweets(10)
    tweet_tensors = process_tweets(tweets)

    predictions = model.predict(tweet_tensors, batch_size=batch_size)
    topic_idx = topics.index(topic) if topic in topics else None
    indices = np.where(predictions == topic_idx)[0]

    return {'tweets': [tweets[i] for i in indices]}


def get_top_categories(n):
    tweets = get_tweets(10)
    tweet_tensors = process_tweets(tweets)

    predictions = model.predict(tweet_tensors, batch_size=batch_size)
    predictions = [topics[prediction] for prediction in predictions]

    counts = Counter(predictions)
    top_topics = counts.most_common(n)

    return [{'topic': topic, 'count': count} for topic, count in top_topics]
