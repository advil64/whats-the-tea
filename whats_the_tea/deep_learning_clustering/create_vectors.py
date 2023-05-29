from pyspark.sql import functions as F
from pyspark.sql import SparkSession
import findspark
import spacy

nlp = spacy.load('en_core_web_lg')

# Initialize Spark
findspark.init()
spark = SparkSession.builder \
    .master('local[*]') \
    .appName('explore') \
    .config('spark.driver.maxResultSize', '8g') \
    .getOrCreate()

# Read JSON data into DataFrame
df = spark.read.json('/common/users/shared/cs543_fall22_group3/combined/combined_raw')


@F.udf
def vectorize(text):
    return nlp(str(text)).vector


# Process DataFrame and add vector column
processed_df = df.withColumn('vector', vectorize(F.col('selected_text')))

# Write processed DataFrame to JSON
processed_df.write.mode('Overwrite').json('/common/users/shared/cs543_fall22_group3/combined/deep_vectors')
