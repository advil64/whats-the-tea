from collections import Counter
from model import load_model
from .config import Config
import json
import numpy as np
import re
import spacy
import string
import torch
import tweepy

nlp = spacy.load('en_core_web_lg')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = load_model('model.pt')
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
    def clean_tweet(tweet):
        cleaned_tweet = re.sub(pattern, '', tweet).lower()
        doc = nlp(cleaned_tweet)
        cleaned_tweet = ' '.join([token.text for token in doc if not token.is_stop and len(token.text) > 2])
        cleaned_tweet = ''.join([char for char in cleaned_tweet if char not in string.punctuation])

        return cleaned_tweet

    pattern = re.compile(
        r'https?:\/\/(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&\/=]*)')

    cleaned_tweets = [clean_tweet(tweet) for tweet in tweets]
    tokenized_tweets = np.array([nlp(tweet).vector for tweet in cleaned_tweets])
    tweet_tensors = torch.tensor(tokenized_tweets).unsqueeze(1)

    return tweet_tensors


def filter_tweets(topic):
    tweets = get_tweets(10)
    tweet_tensors = process_tweets(tweets)

    predictions = model.predict(tweet_tensors, batch_size=batch_size)
    topic_idx = topics.index(topic) if topic in topics else None
    indices = np.where(predictions == topic_idx)[0]

    return {'tweets': [tweets[i] for i in indices]}


def get_top_topics(n):
    tweets = get_tweets(10)
    tweet_tensors = process_tweets(tweets)

    predictions = model.predict(tweet_tensors, batch_size=batch_size)
    predictions = [topics[prediction] for prediction in predictions]

    counts = Counter(predictions)
    top_topics = counts.most_common(n)

    return [{'topic': topic, 'count': count} for topic, count in top_topics]
