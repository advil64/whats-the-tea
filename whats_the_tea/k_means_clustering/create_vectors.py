from pyspark.ml.feature import Word2Vec, Word2VecModel
from pyspark.ml.linalg import Vectors
from pyspark.sql import SparkSession
import findspark

# Initialize SparkSession
findspark.init()
spark = SparkSession.builder \
    .master('local[*]') \
    .appName('word2vec') \
    .config('spark.driver.maxResultSize', '8g') \
    .getOrCreate()

# Load Word2Vec model and vector table
model_path = '/common/users/shared/cs543_fall22_group3/models/word2vec'
model = Word2VecModel.load(model_path)
vector_table = model.getVectors()

# Read processed dataframe
processed_df = spark.read.json('/common/users/shared/cs543_fall22_group3/combined/combined_processed')

# Write tokens to text file
tokens_output_path = '/common/users/shared/cs543_fall22_group3/combined/tokens.txt'
processed_df.drop('selected_text') \
    .coalesce(1) \
    .write.format('text') \
    .option('header', 'false') \
    .mode('overwrite') \
    .save(tokens_output_path)

# Process tokens and calculate vectors
vectors_output_path = '/common/users/shared/cs543_fall22_group3/combined/combined_vectors'
vectors_df = None
count = 0

with open(tokens_output_path) as file:
    for line in file:
        if count > 1000:
            break
        count += 1
        print(f'Finished calculating {count} vectors')

        tokens = line.strip().split(',')
        filtered_vectors = vector_table.filter(vector_table.word.isin(tokens)).drop('word').collect()
        features = Vectors.dense([v[0][0] for v in filtered_vectors])
        weighCol = 1.0
        temp_df = spark.createDataFrame([(features, weighCol)], ["features", "weighCol"])

        if vectors_df is None:
            vectors_df = temp_df
        else:
            vectors_df = vectors_df.union(temp_df)

# Write vectors to JSON
vectors_df.write.mode('Overwrite').json('/common/users/shared/cs543_fall22_group3/combined/combined_vectors')
