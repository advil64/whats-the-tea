import tweepy
from dotenv import load_dotenv
import os
import pandas as pd
import numpy as np
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import spacy


"""
Prepping dependencies
"""
load_dotenv()
nlp = spacy.load("en_core_web_lg")

"""
load pytorch model
NOTE: model not trained yet
"""

"""
Getting twitter credentials from environment variables and instantiating a tweepy instance
"""

auth = tweepy.OAuthHandler(os.environ.get("consumer_key"), os.environ.get("consumer_secret"))
auth.set_access_token(os.environ.get("access_token"), os.environ.get("access_token_secret"))

api = tweepy.API(auth)

"""
Text preprocessing
"""
def preprocess(text):
    return nlp(text).vector

"""
Uses tweepy to get information about the twitter user
"""

def getTwitterInfo(username):
    return api.get_user(screen_name=username)

def getTwitterTimeline(username):
    topics = {}
    timeline = api.user_timeline(screen_name=username, count=100)
    tweets_for_csv = [tweet.text for tweet in timeline]
    # Traverse through the first 100 tweets
    for latest_tweet in tweets_for_csv:
        bow_vector = dictionary.doc2bow(preprocess(latest_tweet))
        for index, score in sorted(lda_model[bow_vector], key=lambda tup: -1 * tup[1]):
            topic = lda_model.print_topic(index, 5)
            if topic in topics:
                topics.update({topic: topics.get(topic) + score.item()})
            else:
                topics.update({topic: score.item()})
    sorted_topics = {k: v for k, v in sorted(topics.items(), key=lambda item: item[1], reverse=True)}
    return sorted_topics