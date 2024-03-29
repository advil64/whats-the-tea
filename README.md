# What's the Tea
![Web Application](deep_learning_clustering/deliverables/app.png)

## Abstract
We have implemented Topic Classification for news articles to classify different articles into multiple topics in real-time. We have used a deep learning network model to classify news articles into 42 categories. We trained our classification model to classify different news articles, and then applied this model to real-time Tweets from various authorized Twitter news handles to predict the topics at any given time. We also allow users to view the top ’N’ most popular Twitter topics at any given time and see their related Tweets as well.

**View the report [here](https://github.com/advil64/whats-the-tea/blob/main/whats_the_tea/deep_learning_clustering/deliverables/What_s_the_Tea__Topic_Classification_of_News_Articles_Deep_Learning.pdf).**

## Data Source
We are using multiple datasets for this project:

**HuffPost Dataset**: https://www.kaggle.com/datasets/rmisra/news-category-dataset \
**RealNews Dataset**: https://paperswithcode.com/dataset/realnews \
**News Aggregator Dataset**: https://www.kaggle.com/datasets/uciml/news-aggregator-dataset \
**A Million News Headlines**: https://www.kaggle.com/datasets/therohk/million-headlines \
**All the News 2.0**: https://components.one/datasets/all-the-news-2-news-articles-dataset/ \
**India News Headlines Dataset**: https://www.kaggle.com/datasets/therohk/india-headlines-news-dataset

## Installation
- Create a `.env` file in the root directory with the following fields for Tweepy user authentication:
```
bearer_token=YOUR_BEARER_TOKEN
consumer_key=YOUR_CONSUMER_KEY
consumer_secret=YOUR_CONSUMER_SECRET
access_token=YOUR_ACCESS_TOKEN
access_token_secret=YOUR_ACCESS_TOKEN_SECRET
```
- Install required libraries: `pip install -r requirements.txt`
- In the `deep_learning_clustering/twitter_dash` directory, run the server: `flask --app main.py run`
- Open a web browser and visit the following URL: `http://127.0.0.1:5000/api/docs`

## Contributors
<table>
  <tr>
    <td align="center"><a href="https://github.com/advil64"><img src="https://avatars.githubusercontent.com/u/15657337?v=4" width="100px;" alt=""/><br /><sub><b>Advith Chegu</b></sub></a></td>
    <td align="center"><a href="https://github.com/Vipul97"><img src="https://avatars.githubusercontent.com/u/16150834?v=4" width="100px;" alt=""/><br /><sub><b>Vipul Gharde</b></sub></a><br /></td>
    <td align="center"><a href="https://github.com/dikshawuthoo"><img src="https://avatars.githubusercontent.com/u/92066985?v=4" width="100px;" alt=""/><br /><sub><b>Diksha Wuthoo</b></sub></a><br /></td>
  </tr>
</table>
