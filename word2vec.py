# %%
import findspark
findspark.init()

from pyspark.ml.feature import Word2Vec
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext

from pyspark import SparkContext
from pyspark.streaming import StreamingContext #Import streaming context
from pyspark.sql import SparkSession

# %%
spark = SparkSession.builder\
    .master('local[*]')\
    .appName('word2vec')\
    .getOrCreate()
sc = spark.sparkContext._conf.setAll([('spark.driver.maxResultSize', '8g')])

# %%
df = spark.read.json('/common/users/shared/cs543_fall22_group3/combined/combined_processed')

# %%
word2vec = Word2Vec(vectorSize=1, seed=42, minCount=0, inputCol = 'cleaned_text', outputCol = 'output_vectors')

# %%
model = word2vec.fit(df)

# %%
model.write().overwrite().save('/common/users/shared/cs543_fall22_group3/models/word2vec')