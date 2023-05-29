from flask import Flask, request, Blueprint
from flask_cors import CORS
from flask_restx import Resource, Api, fields
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

GET_TWEETS = api.model('TweetInfo', {
    'tweets': fields.List(fields.String())
})

GET_CATEGORIES = api.model('TweetInfo', {
    'categories': fields.List(fields.String()),
    'counts': fields.List(fields.Integer())
})

# Documentation for Swagger UI
ns_tweets = api.namespace(
    'tweets', description='Gets the Tweets from news-related accounts'
)

ns_categories = api.namespace(
    'categories', description='Gets the top n categories from the fetched Tweets'
)


@ns_tweets.route('')
class GetTweets(Resource):
    '''
    Returns Tweets from news-related accounts
    '''

    @api.param(
        'Topic',
        description='The news topic you would like to get the Tweets for.',
        type='string',
    )
    @api.marshal_with(GET_TWEETS, mask=None)
    def get(self):
        topic = request.args.get('Topic')
        filtered_tweets = filter_tweets(topic)

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
        n = request.args.get('n')
        categories, counts = get_top_categories(n)

        return {
            'categories': categories,
            'counts': counts
        }
