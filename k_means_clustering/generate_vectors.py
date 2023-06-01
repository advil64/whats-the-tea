from pyspark.ml.feature import Word2VecModel
from pyspark.sql import functions as F
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .master('local[*]') \
    .appName('generate_vectors') \
    .config('spark.driver.maxResultSize', '8g') \
    .getOrCreate()

ROOT = '../'

processed_df = spark.read.json(f'../{ROOT}/dataset/combined_processed')
processed_df = processed_df.withColumn('article', F.split(F.col('article'), ','))

word2vec_model = Word2VecModel.load(f'{ROOT}/models/word2vec')
vectorized_df = word2vec_model.transform(processed_df.select('article'))

vectorized_df.write.mode('overwrite').json(f'{ROOT}/combined/combined_vectors')
