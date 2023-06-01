from pyspark.ml.feature import Word2Vec
from pyspark.sql import SparkSession

# Initialize Spark
spark = SparkSession.builder \
    .master('local[*]') \
    .appName('train_word2vec') \
    .config('spark.driver.maxResultSize', '8g') \
    .getOrCreate()

ROOT = '../'

# Read JSON data
processed_df = spark.read.json(f'{ROOT}/dataset/combined/combined_processed')

# Configure Word2Vec
word2vec = Word2Vec(vectorSize=1, seed=42, minCount=0, inputCol='article', outputCol='article_embedding')

# Train Word2Vec model
model = word2vec.fit(processed_df)

# Save the trained model
model.write().overwrite().save(f'{ROOT}/models/word2vec')
