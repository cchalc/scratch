# Databricks notebook source
# MAGIC %md #### Refrences
# MAGIC - https://blog.vlgdata.io/post/nlp_hugging_face/
# MAGIC - https://github.com/datatrigger/nlp_hugging_face/blob/main/bbc_news.ipynb
# MAGIC - https://www.kaggle.com/competitions/learn-ai-bbc/data

# COMMAND ----------

# MAGIC %md ### Set up data

# COMMAND ----------

spark.sql("use catalog cjc")
spark.sql("use schema scratch")

# COMMAND ----------

import pandas as pd
import os

os.environ['TRANSFORMERS_CACHE'] = '/Volumes/cjc/scratch/llm_cache'

# COMMAND ----------

#spark.conf.set("spark.sql.files.maxPartitionBytes", 250000) #~2MB/8
spark.conf.set("spark.sql.files.maxPartitionBytes", 285714) #~2MB/7

# COMMAND ----------

# dataset is only 1.9MiB
df = spark.read.table("bbc_news_train")
display(df)

# COMMAND ----------

num_partitions = df.rdd.getNumPartitions()
print("Number of partitions in the DataFrame:", num_partitions)


# COMMAND ----------

candidate_labels = df.select('Category').distinct().rdd.flatMap(lambda x: x).collect()
# print(labels)
# ['sport', 'politics', 'entertainment', 'business', 'tech']

# COMMAND ----------

from pyspark.sql.functions import pandas_udf, StringType
from transformers import pipeline
import pandas as pd
import torch

# COMMAND ----------

# Use GPU if available
device = 0 if torch.cuda.is_available() else -1

# Define the zero-shot classification model
zero_shot_classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device=device
)

# COMMAND ----------

broadcast_pipeline = spark.sparkContext.broadcast(zero_shot_classifier)

# COMMAND ----------

# Define a Pandas UDF to perform zero-shot classification
@pandas_udf('string')
def classify_text(texts: pd.Series) -> pd.Series:
    pipe = broadcast_pipeline.value(
      ("classify: " + texts).to_list()
                      ,candidate_labels
                      ,batch_size=8)
    labels = [result['labels'][0] for result in pipe]
    return pd.Series(labels)

# COMMAND ----------

# MAGIC %md ### Run on a sample

# COMMAND ----------

# df.limit(200).coalesce(8).rdd.getNumPartitions()

# COMMAND ----------

df_200 = df.limit(200)
# since coalesce is only supposed to decrease the number of partitions, this doesn't do anything.
# using repartition in the next cell
# df_200.coalesce(8)

# COMMAND ----------

df_200 = df.limit(200).repartition(8)

# COMMAND ----------

sc.setJobDescription("200 classification")
result_df = (df_200
             .withColumn(
               "label_pred_zero_shot", classify_text(df.Text)
               )
             )
display(result_df)

# COMMAND ----------

# MAGIC %md #### Issue
# MAGIC Cannot distribute the job properly. There is no shuffle so the data was partitioned on read into 9 partitions using `spark.conf.set("spark.sql.files.maxPartitionBytes", 250000) #~2MB/8`. Trying to run a test on a 200 row dataframe. What I think should happen: dataset is split into 8 and then the udf `classify_text` should pick up the column as a pandas.Series and return the labels. 

# COMMAND ----------

# MAGIC %md
# MAGIC #### Update - DP
# MAGIC
# MAGIC Was able to distribute primary dataframe to 8 partitions by using `spark.conf.set("spark.sql.files.maxPartitionBytes", 285714) #~2MB/7`. Needed to use `repartition(8)` on `df_200` to cause a shuffle. However, this is still hanging during the shuffle stage. Looking at certain tasks shows an error message:
# MAGIC ```
# MAGIC 23/10/23 14:18:34 WARN HangingTaskDetector: Task 49 is probably not making progress because its metrics (Map(internal.metrics.shuffle.read.localBlocksFetched -> 0, internal.metrics.shuffle.read.remoteBytesReadToDisk -> 0, internal.metrics.shuffle.write.bytesWritten -> 0, internal.metrics.output.recordsWritten -> 0, internal.metrics.shuffle.write.recordsWritten -> 0, internal.metrics.memoryBytesSpilled -> 0, internal.metrics.shuffle.read.remoteBytesRead -> 35852, internal.metrics.diskBytesSpilled -> 0, internal.metrics.shuffle.read.localBytesRead -> 0, internal.metrics.shuffle.read.recordsRead -> 25, internal.metrics.output.bytesWritten -> 0, internal.metrics.input.bytesRead -> 0, internal.metrics.input.recordsRead -> 0, internal.metrics.shuffle.read.remoteBlocksFetched -> 1)) has not changed since Mon Oct 23 13:58:34 UTC 2023
# MAGIC ```
# MAGIC
# MAGIC Process eventually finished after 44 minutes.

# COMMAND ----------

# MAGIC %md ### Summary
# MAGIC #### Single node GPU
# MAGIC **Specs**
# MAGIC ```
# MAGIC Summary
# MAGIC 1 Driver 16 GB Memory, 4 Cores
# MAGIC Runtime 14.1.x-gpu-ml-scala2.12
# MAGIC Unity Catalog g4dn.xlarge 0.71 DBU/h
# MAGIC ```
# MAGIC
# MAGIC **Runs**
# MAGIC - Single Classification (1): Command took 32.20 seconds
# MAGIC - Small Sample (10): Command took 37.62 seconds
# MAGIC - Larger Sample (200, batch size 8): Command took 3.76 minutes
# MAGIC - Full dataset (1490, batch size 8): Command took 25.26 minutes
# MAGIC
# MAGIC #### 2 worker CPU
# MAGIC **Specs**
# MAGIC ```
# MAGIC Summary
# MAGIC 2 Workers 61 GB Memory 8 Cores 
# MAGIC 1 Driver 30.5 GB Memory, 4 Cores
# MAGIC Runtime 14.1.x-cpu-ml-scala2.12
# MAGIC Unity Catalog i3.xlarge 3 DBU/h
# MAGIC ```
# MAGIC
# MAGIC **Runs**
# MAGIC - Larger Sample (200, batch size 8): Command took 44 minutes
# MAGIC - Full dataset (1490, batch size 8): Command took ____

# COMMAND ----------

a = (3.76/60) * 0.71
print(a)

# COMMAND ----------

b = (44/60)*3
print(b)

# COMMAND ----------

b/a

# COMMAND ----------

3.76/60
