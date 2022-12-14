# %%
import findspark
findspark.init()

from pyspark.ml.feature import Word2Vec
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext

from pyspark import SparkContext
from pyspark.streaming import StreamingContext #Import streaming context
from pyspark.sql import SparkSession

import spacy
from pyspark.sql import functions as F

# %%
nlp = spacy.load('en_core_web_lg')

# %%
spark = SparkSession.builder\
    .master('local[*]')\
    .appName('explore')\
    .getOrCreate()
sc = spark.sparkContext._conf.setAll([('spark.driver.maxResultSize', '8g')])

# %%
df = spark.read.json('/common/users/shared/cs543_fall22_group3/combined/combined_raw')

def vectorize(text):
    return nlp(str(text)).vector

vectorize_udf = F.udf(lambda z: vectorize(z))
processed_df = df.withColumn("vector", vectorize_udf(F.col("selected_text")))

# %%
processed_df.write.mode("Overwrite").json('/common/users/shared/cs543_fall22_group3/combined/deep_vectors')

#%%