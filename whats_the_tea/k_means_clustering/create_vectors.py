# %%
import findspark
findspark.init()

from pyspark.ml.feature import Word2Vec, Word2VecModel
from pyspark.sql import functions as F
import numpy as np
from pyspark.ml.linalg import Vectors

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
model = Word2VecModel.load('/common/users/shared/cs543_fall22_group3/models/word2vec')
vector_table = model.getVectors()

# %%
processed_df = spark.read.json('/common/users/shared/cs543_fall22_group3/combined/combined_processed')

# %%
processed_df.drop('selected_text').coalesce(1).write.format('text').option('header', 'false').mode('overwrite').save('/common/users/shared/cs543_fall22_group3/combined/tokens.txt')

# %%
vectors_df = None
count = 0

for line in open('/common/users/shared/cs543_fall22_group3/combined/tokens/sentences.txt'):

    if count > 1000:
        break
    count += 1
    print('Finished calculating {} vectors'.format(count))
    
    tokens = line.strip().split(',')
    vector_list = vector_table.filter(vector_table.word.isin(tokens)).drop('word').collect()
    if vectors_df is None:
        vectors_df = spark.createDataFrame([(Vectors.dense([v[0][0] for v in vector_list]), 1.0)], ["features", "weighCol"])
    else:
        temp = spark.createDataFrame([(Vectors.dense([v[0][0] for v in vector_list]), 1.0)], ["features", "weighCol"])
        vectors_df = vectors_df.union(temp)

# %%
vectors_df.write.mode("Overwrite").json('/common/users/shared/cs543_fall22_group3/combined/combined_vectors')
