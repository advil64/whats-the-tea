from flask import Flask, request, Blueprint
from flask_cors import CORS
from flask_restx import Resource, Api, fields
from torch.utils.data import DataLoader
from torchtext.data.functional import to_map_style_dataset

from .query import *

app = Flask(__name__)
cors = CORS(app)

blueprint = Blueprint('api', __name__, url_prefix='/api')

api = Api(
    blueprint,
    version='0.1.0',
    title='Twitter Dashboard API',
    doc='/docs'
)

app.register_blueprint(blueprint)

# GET_TWEETS = api.model('UserInfo', {
#     'username': fields.String(),
#     'profile_url': fields.String(),
#     'bio': fields.String(),
#     'date_created': fields.DateTime(dt_format='iso8601'),
#     'display_name': fields.String(),
#     'vectors': fields.List(fields.String()),
#     'scores': fields.List(fields.Float())})

GET_TWEETS = api.model('TweetInfo', {
    'tweets': fields.List(fields.String())
})

GET_CATEGORIES = api.model('TweetInfo', {
    'categories': fields.List(fields.String()),
    'counts': fields.List(fields.Integer())
})

# documentation for swagger UI
ns_tweets = api.namespace(
    'tweets', description='Gets the Tweets from news-related accounts'
)

ns_categories = api.namespace(
    'categories', description='Gets the top n categories from the fetched Tweets'
)


@ns_tweets.route('')
class GetUserAnalytics(Resource):
    '''
    Returns Tweets from news related accounts
    '''

    @api.param(
        'Topic',
        description='The news topic you would like to get the Tweets for.',
        type='string',
    )
    @api.marshal_with(GET_TWEETS, mask=None)
    def get(self):
        tweets = get_tweets(n=10, batch_size=64)
        embeddings = get_embeddings(tweets)
        df = make_dataframe(tweets, embeddings)

        classify_dataset = to_map_style_dataset(df['vector'])
        test_dataloader = DataLoader(classify_dataset, batch_size=64, shuffle=True, collate_fn=collate_batch)

        predictions = predict(test_dataloader)
        df['label'] = predictions

        topic = request.args.get('Topic')
        filtered_tweets = filter_tweets(df, topic)

        return {'tweets': filtered_tweets}


@ns_categories.route('')
class GetCategories(Resource):
    '''
    Returns the Top n Categories from the fetched Tweets
    '''

    @api.param(
        'n',
        description='The number of top categories to return',
        type='int',
    )
    @api.marshal_with(GET_CATEGORIES, mask=None)
    def get(self):
        tweets = get_tweets(n=10, batch_size=64)
        embeddings = get_embeddings(tweets)
        df = make_dataframe(tweets, embeddings)

        classify_dataset = to_map_style_dataset(df['vector'])
        test_dataloader = DataLoader(classify_dataset, batch_size=64, shuffle=True, collate_fn=collate_batch)

        predictions = predict(test_dataloader)
        df['label'] = predictions

        n = request.args.get('n')
        categories, counts = get_top_categories(df, n)

        return {
            'categories': categories,
            'counts': counts
        }
