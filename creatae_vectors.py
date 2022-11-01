# %%
from pyspark.ml.feature import Word2Vec, Word2VecModel

# %%
model = Word2VecModel.load('/common/users/shared/cs543_fall22_group3/models/word2vec')
vector_table = model.getVectors()
vector_table.createOrReplaceTempView("vector_table")

# %%
processed_df = spark.read.json('/common/users/shared/cs543_fall22_group3/combined/combined_processed')

# %%
# find the corresponding vectors for some row
def calculate_vec(line):
    vecs = [list(vector_table.where(vector_table.word == 'sezwho').collect()[0])[1][0] for w in line]
    return ",".join(vecs)

calculate_vec_udf = F.udf(lambda z: calculate_vec(z))
vectorized_df = processed_df.withColumn("vectors", F.split(calculate_vec_udf(F.col("cleaned_text")), ","))

