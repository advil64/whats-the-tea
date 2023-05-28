from pyspark.ml.feature import Word2Vec
from pyspark.sql import SparkSession
import findspark

# Initialize Spark
findspark.init()
spark = SparkSession.builder \
    .master('local[*]') \
    .appName('word2vec') \
    .config('spark.driver.maxResultSize', '8g') \
    .getOrCreate()

# Read JSON data
df = spark.read.json('/common/users/shared/cs543_fall22_group3/combined/combined_processed')

# Configure Word2Vec
word2vec = Word2Vec(vectorSize=1, seed=42, minCount=0, inputCol='cleaned_text', outputCol='output_vectors')

# Train Word2Vec model
model = word2vec.fit(df)

# Save the trained model
model.write().overwrite().save('/common/users/shared/cs543_fall22_group3/models/word2vec')
