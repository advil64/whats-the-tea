from dotenv import load_dotenv
from torch.utils.data import DataLoader
from torchtext.data.functional import to_map_style_dataset
import json
import math
import os
import pandas as pd
import spacy
import torch
import torch.nn as nn
import tweepy

load_dotenv()
nlp = spacy.load('en_core_web_lg')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

bearer_token = os.environ.get('bearer_token')
consumer_key = os.environ.get('consumer_key')
consumer_secret = os.environ.get('consumer_secret')
access_token = os.environ.get('access_token')
access_token_secret = os.environ.get('access_token_secret')

client = tweepy.Client(
    bearer_token=bearer_token,
    consumer_key=consumer_key,
    consumer_secret=consumer_secret,
    access_token=access_token,
    access_token_secret=access_token_secret,
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


class TextClassificationModel(nn.Module):
    def __init__(self, num_classes=42, embed_dim=1, vocab_size=45, pad_index=0,
                 stride=1, kernel_size=3, conv_out_size=64, dropout_rate=0.25):
        super(TextClassificationModel, self).__init__()

        # Embedding layer parameters
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.pad_index = pad_index

        # Conv layer parameters
        self.stride = stride
        self.kernel_size = kernel_size
        self.conv_out_size = conv_out_size

        # Misc
        self.dropout_rate = dropout_rate

        # Layers
        self.conv = torch.nn.Conv1d(self.embed_dim, self.conv_out_size, self.kernel_size, self.stride)
        self.hidden_act = torch.relu
        self.max_pool = torch.nn.MaxPool1d(self.kernel_size, self.stride)

        self.flatten = lambda x: x.view(x.shape[0], x.shape[1] * x.shape[2])

        self.fc = torch.nn.Linear(self._linear_layer_in_size(), self.num_classes)

        if self.dropout_rate:
            self.dropout = torch.nn.Dropout(self.dropout_rate)

    def _linear_layer_in_size(self):
        out_conv_1 = ((self.embed_dim - 1 * (self.kernel_size - 1) - 1) / self.stride) + 1
        out_conv_1 = math.floor(out_conv_1)
        out_pool_1 = ((out_conv_1 - 1 * (self.kernel_size - 1) - 1) / self.stride) + 1
        out_pool_1 = math.floor(out_pool_1)

        return 18944  # out_pool_1 * self.conv_out_size

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        x = self.conv(x)
        x = self.hidden_act(x)
        x = self.max_pool(x)
        x = self.flatten(x)

        if self.dropout_rate:
            x = self.dropout(x)

        x = self.fc(x)
        return x


model = TextClassificationModel()
model.load_state_dict(torch.load('class_model.pt', map_location=device))
model.to(torch.device(device))


def predict(dataloader):
    model.eval()
    pred = []

    with torch.no_grad():
        for idx, vector in enumerate(dataloader):
            predicted_label = model(vector)
            pred.extend(predicted_label.argmax(1).cpu().detach().numpy())

    return pred


def collate_batch(batch):
    embedding_list = [torch.tensor(_embedding, dtype=torch.float32) for _embedding in batch]
    embedding_list = torch.stack(embedding_list)

    return embedding_list.to(device)


def filter_tweets(df, topic):
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

    filtered_df = df.loc[df['label'] == label_mapping[topic]]

    return filtered_df['text'].values


def get_top_categories(df, n):
    label_mapping = {
        0: 'ARTS',
        1: 'ARTS & CULTURE',
        2: 'BLACK VOICES',
        3: 'BUSINESS',
        4: 'COLLEGE',
        5: 'COMEDY',
        6: 'CRIME',
        7: 'CULTURE & ARTS',
        8: 'DIVORCE',
        9: 'EDUCATION',
        10: 'ENTERTAINMENT',
        11: 'ENVIRONMENT',
        12: 'FIFTY',
        13: 'FOOD & DRINK',
        14: 'GOOD NEWS',
        15: 'GREEN',
        16: 'HEALTHY LIVING',
        17: 'HOME & LIVING',
        18: 'IMPACT',
        19: 'LATINO VOICES',
        20: 'MEDIA',
        21: 'MONEY',
        22: 'PARENTING',
        23: 'PARENTS',
        24: 'POLITICS',
        25: 'QUEER VOICES',
        26: 'RELIGION',
        27: 'SCIENCE',
        28: 'SPORTS',
        29: 'STYLE',
        30: 'STYLE & BEAUTY',
        31: 'TASTE',
        32: 'TECH',
        33: 'THE WORLDPOST',
        34: 'TRAVEL',
        35: 'U.S. NEWS',
        36: 'WEDDINGS',
        37: 'WEIRD NEWS',
        38: 'WELLNESS',
        39: 'WOMEN',
        40: 'WORLD NEWS',
        41: 'WORLDPOST'
    }

    df['label'] = df['label'].apply(lambda x: label_mapping[x])
    df = df.groupby('label')['label'].count().reset_index(name='count').sort_values(['count'], ascending=False)
    df = df.head(int(n))

    categories = df['label'].values.tolist()
    counts = df['count'].values.tolist()

    return categories, counts
