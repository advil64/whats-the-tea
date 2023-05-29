from dotenv import load_dotenv
import os

load_dotenv()


class Config:
    BEARER_TOKEN = os.environ.get('bearer_token')
    CONSUMER_KEY = os.environ.get('consumer_key')
    CONSUMER_SECRET = os.environ.get('consumer_secret')
    ACCESS_TOKEN = os.environ.get('access_token')
    ACCESS_TOKEN_SECRET = os.environ.get('access_token_secret')
