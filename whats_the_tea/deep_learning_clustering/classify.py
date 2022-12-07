import findspark
findspark.init()

from pyspark.ml.feature import Word2Vec
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext

from pyspark import SparkContext
from pyspark.streaming import StreamingContext #Import streaming context
from pyspark.sql import SparkSession

import spacy_sentence_bert
from pyspark.sql import functions as F

spark = SparkSession.builder\
    .master('local[*]')\
    .appName('explore')\
    .getOrCreate()
sc = spark.sparkContext._conf.setAll([('spark.driver.maxResultSize', '8g')])

df = spark.read.json('/common/users/shared/cs543_fall22_group3/combined/deep_vectors')
